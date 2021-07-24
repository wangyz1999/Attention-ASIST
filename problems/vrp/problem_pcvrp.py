from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.vrp.state_pcvrp import StatePCVRP
from problems.vrp.state_sdvrp import StateSDVRP
from utils.beam_search import beam_search


class PCVRP(object):

    NAME = 'pcvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    PRICE_MODE = 1
    DISCOUNT_FACTOR = 0.9
    high_value = 1
    player_role = 'medic'
    medic_model = None
    medic_graph_size = 55
    rubble_graph_size = 23
    medic_tool_durability = 20
    engineer_tool_durability = 131
    high_value_victim_size = 5
    medic_speed = 0.007
    engineer_speed = 0.005
    green_triage_time = 7.5
    yellow_triage_time = 15
    break_rubble_time = 0.5

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        prize = dataset['prize']
        prize_with_depot = torch.cat(
            (
                torch.zeros_like(prize[:, :1]),
                prize
            ),
            1
        )

        p = prize_with_depot.gather(1, pi)

        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], - PCVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= PCVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        # length = (
        #     (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
        #     + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
        #     + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        # ), None

        length = (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        )

        # minimize the sum of total length and negative prize
        # print(p.sum(-1))

        if PCVRP.player_role == 'medic':
            if PCVRP.PRICE_MODE == 1:
                discount_factor = PCVRP.DISCOUNT_FACTOR
                discount_factor_list = [1]
                for dfl in range(p.size()[1] - 1):
                    discount_factor_list.append(discount_factor_list[-1] * discount_factor)
                p_discount_matrix = torch.Tensor(discount_factor_list)
                p_discount_matrix = p_discount_matrix.to(device=p.device)
                p_discount = p_discount_matrix[None, :] * p

            if PCVRP.PRICE_MODE == 2:
                stepwise_discount_factor = PCVRP.DISCOUNT_FACTOR
                stepwise_discount_factors = []
                for run in pi:
                    curr_factor = 1
                    stepwise_discount_factor_list = []
                    for pi_step in run:
                        if pi_step == 0:
                            curr_factor *= stepwise_discount_factor
                            stepwise_discount_factor_list.append(curr_factor)
                        else:
                            stepwise_discount_factor_list.append(curr_factor)
                    stepwise_discount_factors.append(stepwise_discount_factor_list)
                stepwise_discount_factors = torch.Tensor(stepwise_discount_factors)
                stepwise_discount_factors = stepwise_discount_factors.to(device=p.device)
                p_stepwise_discount = stepwise_discount_factors * p


            if PCVRP.PRICE_MODE == 0:
                return -p.sum(-1), None
            elif PCVRP.PRICE_MODE == 1:
                return length - p_discount.sum(-1), None
            elif PCVRP.PRICE_MODE == 2:
                return length - p_stepwise_discount.sum(-1), None

        elif PCVRP.player_role == 'engineer':
            medic_tour = dataset['medic_tour']
            medic_tour = medic_tour.type(torch.int64)
            engineer_tour = pi.data

            medic_loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['victim_loc']), 1)
            medic_d = medic_loc_with_depot.gather(1, medic_tour[..., None].expand(*medic_tour.size(), medic_loc_with_depot.size(-1)))

            medic_length = torch.cat(
                (
                    (medic_d[:, 0] - dataset['depot']).norm(p=2, dim=1).unsqueeze(1),
                    (medic_d[:, 1:] - medic_d[:, :-1]).norm(p=2, dim=2),
                    (medic_d[:, -1] - dataset['depot']).norm(p=2, dim=1).unsqueeze(1)
                ), 1
            )

            medic_tour_add_zero = torch.cat((medic_tour, torch.zeros(medic_tour.size(0), 1)), 1)
            medic_move_time = medic_length / PCVRP.medic_speed
            medic_triage_time = torch.zeros_like(medic_move_time)
            medic_triage_time[medic_tour_add_zero > PCVRP.medic_graph_size - PCVRP.high_value_victim_size] = PCVRP.yellow_triage_time
            medic_triage_time[medic_tour_add_zero <= PCVRP.medic_graph_size - PCVRP.high_value_victim_size] = PCVRP.green_triage_time
            medic_triage_time[medic_tour_add_zero == 0] = 0
            medic_time = medic_move_time + medic_triage_time

            medic_cumulative_time = torch.zeros_like(medic_time)
            medic_sum_time = torch.zeros((medic_time.size(0)))
            for step in range(medic_time.size(1)):
                medic_sum_time += medic_time[:, step]
                medic_cumulative_time[:, step] = medic_sum_time

            engineer_loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
            engineer_d = engineer_loc_with_depot.gather(1, engineer_tour[..., None].expand(*engineer_tour.size(),
                                                                                  engineer_loc_with_depot.size(-1)))

            engineer_length = torch.cat(
                (
                    (engineer_d[:, 0] - dataset['depot']).norm(p=2, dim=1).unsqueeze(1),
                    (engineer_d[:, 1:] - engineer_d[:, :-1]).norm(p=2, dim=2),
                    (engineer_d[:, -1] - dataset['depot']).norm(p=2, dim=1).unsqueeze(1)
                ), 1
            )

            engineer_tour_add_zero = torch.cat((engineer_tour, torch.zeros(engineer_tour.size(0), 1)), 1)
            engineer_move_time = engineer_length / PCVRP.engineer_speed
            engineer_rubble_time = torch.zeros_like(engineer_move_time)
            engineer_rubble_time[engineer_tour_add_zero <= PCVRP.rubble_graph_size] = PCVRP.break_rubble_time
            engineer_time = engineer_move_time + engineer_rubble_time

            engineer_cumulative_time = torch.zeros_like(engineer_time)
            engineer_sum_time = torch.zeros((engineer_time.size(0)))
            for step in range(engineer_time.size(1)):
                engineer_sum_time += engineer_time[:, step]
                engineer_cumulative_time[:, step] = engineer_sum_time

            medic_high_value_time = torch.gather(medic_cumulative_time, 1, (medic_tour_add_zero > PCVRP.medic_graph_size - PCVRP.high_value_victim_size).nonzero()[:, 1].view(-1, PCVRP.high_value_victim_size))
            engineer_high_value_time = torch.gather(engineer_cumulative_time, 1, (engineer_tour_add_zero > PCVRP.rubble_graph_size).nonzero()[:, 1].view(-1, PCVRP.high_value_victim_size))

            high_value_penalty = torch.abs(engineer_high_value_time - medic_high_value_time).sum(1)

            high_value_penalty = high_value_penalty.to(device=p.device)


            # print(dataset)

            return length+high_value_penalty, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PCVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = PCVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, prize, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float),
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'prize': torch.tensor(prize, dtype=torch.float)
    }


class PCVRPDataset(Dataset):
    
    def __init__(self, filename=None, size=28, num_samples=1000000, offset=0, distribution=None):
        assert PCVRP.player_role in ["medic", "engineer"]

        super(PCVRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.
            }

            self.data = []

            for i in range(num_samples):
                # t = torch.ones(size) * 5/55

                if PCVRP.player_role == "medic":
                    prob = torch.ones(size) * 4 / 20
                    prize = torch.bernoulli(prob)
                    prize[prize == 0] = 0.1
                    prize[prize == 1] = PCVRP.high_value
                    self.data.append({
                        'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                        # Uniform 1 - 9, scaled by capacities
                        # 'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                        'demand': torch.ones(size) / PCVRP.medic_tool_durability,
                        'depot': torch.FloatTensor(2).uniform_(0, 1),
                        'prize': prize
                    })
                elif PCVRP.player_role == "engineer":
                    # medic_prob = torch.ones(medic_graph_size) * 4 / 20
                    # medic_prize = torch.bernoulli(medic_prob)
                    medic_prize = torch.ones(PCVRP.medic_graph_size) * 0.1
                    medic_prize[-PCVRP.high_value_victim_size:] = PCVRP.high_value
                    victim_loc = torch.FloatTensor(PCVRP.medic_graph_size, 2).uniform_(0, 1)
                    medic_batch = {
                        'loc': victim_loc.unsqueeze(0),
                        # Uniform 1 - 9, scaled by capacities
                        # 'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                        'demand': (torch.ones(PCVRP.medic_graph_size) / PCVRP.medic_tool_durability).unsqueeze(0),
                        'depot': torch.FloatTensor(2).uniform_(0, 1).unsqueeze(0),
                        'prize': medic_prize.unsqueeze(0)
                    }
                    # prize = torch.cat((torch.ones(medic_graph_size-high_value_victim_size)*0.1, torch.ones(high_value_victim_size)*high_value))

                    PCVRP.player_role = 'medic'
                    PCVRP.medic_model.eval()
                    PCVRP.medic_model.set_decode_type('greedy')
                    with torch.no_grad():
                        length, log_p, pi = PCVRP.medic_model(medic_batch, return_pi=True)
                    medic_tour = pi
                    PCVRP.player_role = 'engineer'


                    rubbles = torch.randint(0, PCVRP.medic_graph_size, (PCVRP.rubble_graph_size,))

                    selected_victim_loc = torch.gather(victim_loc, 0, torch.hstack((rubbles.view(-1, 1), rubbles.view(-1, 1))))
                    high_value_loc = victim_loc[-PCVRP.high_value_victim_size:]
                    # print(len(selected_victim_loc), len(high_value_loc))
                    # print(torch.cat((selected_victim_loc, high_value_loc)).shape)
                    self.data.append({
                        'loc': torch.cat((selected_victim_loc, high_value_loc)),
                        # Uniform 1 - 9, scaled by capacities
                        # 'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                        'medic_tour': medic_tour.squeeze(0),
                        'victim_loc': victim_loc,
                        'demand': torch.ones(size) / PCVRP.engineer_tool_durability,
                        'depot': torch.FloatTensor(2).uniform_(0, 1),
                        'prize': torch.zeros(size)
                    })

        max_tour_length = 0
        for tour in self.data:
            max_tour_length = max(max_tour_length, len(tour['medic_tour']))
        for tour in range(len(self.data)):
            self.data[tour]['medic_tour'] = torch.cat((self.data[tour]['medic_tour'], torch.zeros(max_tour_length-len(self.data[tour]['medic_tour']))))

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
