from torch.utils.data import Dataset
import torch
import os
import pickle
import random
from tqdm import tqdm

from problems.vrp.state_pcvrp import StatePCVRP
from problems.vrp.state_sdvrp import StateSDVRP
from utils.beam_search import beam_search


class PCVRP(object):

    NAME = 'pcvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    PRICE_MODE = 1
    DISCOUNT_FACTOR = 0.9
    HIGH_VALUE = 1
    PLAYER_ROLE = 'engineer'
    MEDIC_MODEL = None
    MEDIC_GRAPH_SIZE = 55
    RUBBLE_GRAPH_SIZE = 23
    MEDIC_TOOL_DURABILITY = 20
    ENGINEER_TOOL_DURABILITY = 131
    HIGH_VALUE_VICTIM_SIZE = 5
    MEDIC_SPEED = 0.007
    ENGINEER_SPEED = 0.005
    GREEN_TRIAGE_TIME = 7.5
    YELLOW_TRIAGE_TIME = 15
    BREAK_RUBBLE_TIME = 0.5
    HIGH_VALUE_MISMATCH_PENALTY_COEFF = 0.004
    LATE_RUBBLE_PENALTY_COEFF = 1.757


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

        if PCVRP.PLAYER_ROLE == 'medic':
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

        elif PCVRP.PLAYER_ROLE == 'engineer':
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

            medic_tour_add_zero = torch.cat((medic_tour, torch.zeros(medic_tour.size(0), 1, device=p.device)), 1)
            medic_move_time = medic_length / PCVRP.MEDIC_SPEED
            medic_triage_time = torch.zeros_like(medic_move_time)
            medic_triage_time[medic_tour_add_zero > PCVRP.MEDIC_GRAPH_SIZE - PCVRP.HIGH_VALUE_VICTIM_SIZE] = PCVRP.YELLOW_TRIAGE_TIME
            medic_triage_time[medic_tour_add_zero <= PCVRP.MEDIC_GRAPH_SIZE - PCVRP.HIGH_VALUE_VICTIM_SIZE] = PCVRP.GREEN_TRIAGE_TIME
            medic_triage_time[medic_tour_add_zero == 0] = 0
            medic_time = medic_move_time + medic_triage_time

            # based on the medic tour, each following element of medic cumulative time represents the current time after going through the node with respestive index
            medic_cumulative_time = torch.zeros_like(medic_time, device=p.device)
            medic_sum_time = torch.zeros(medic_time.size(0), device=p.device)
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

            engineer_tour_add_zero = torch.cat((engineer_tour, torch.zeros(engineer_tour.size(0), 1, device=p.device)), 1)
            engineer_move_time = engineer_length / PCVRP.ENGINEER_SPEED
            engineer_rubble_time = torch.zeros_like(engineer_move_time, device=p.device)
            engineer_rubble_time[engineer_tour_add_zero <= PCVRP.RUBBLE_GRAPH_SIZE] = PCVRP.BREAK_RUBBLE_TIME
            engineer_time = engineer_move_time + engineer_rubble_time

            # based on the engineer tour, each following element of engineer cumulative time represents the current time after going through the node with respestive index
            engineer_cumulative_time = torch.zeros_like(engineer_time, device=p.device)
            engineer_sum_time = torch.zeros(engineer_time.size(0), device=p.device)
            for step in range(engineer_time.size(1)):
                engineer_sum_time += engineer_time[:, step]
                engineer_cumulative_time[:, step] = engineer_sum_time

            medic_high_value_time = torch.gather(medic_cumulative_time, 1, (medic_tour_add_zero > PCVRP.MEDIC_GRAPH_SIZE - PCVRP.HIGH_VALUE_VICTIM_SIZE).nonzero()[:, 1].view(-1, PCVRP.HIGH_VALUE_VICTIM_SIZE))
            engineer_high_value_time = torch.gather(engineer_cumulative_time, 1, (engineer_tour_add_zero > PCVRP.RUBBLE_GRAPH_SIZE).nonzero()[:, 1].view(-1, PCVRP.HIGH_VALUE_VICTIM_SIZE))

            high_value_mismatch_penalty = torch.abs(engineer_high_value_time - medic_high_value_time).sum(1)

            high_value_mismatch_penalty = high_value_mismatch_penalty.to(device=p.device)

            medic_tour_sorted, medic_tour_sorted_indices = torch.sort(medic_tour_add_zero, 1)
            engineer_tour_sorted, engineer_tour_sorted_indices = torch.sort(engineer_tour_add_zero, 1)

            # below represents the time point that visits victim 0,1,2,3,4...
            medic_node_number_order_time = torch.gather(medic_cumulative_time, 1, medic_tour_sorted_indices)
            engineer_node_number_order_time = torch.gather(engineer_cumulative_time, 1, engineer_tour_sorted_indices)

            # medic_rubble_victim_time = torch.gather(medic_cumulative_time, 1, ((medic_tour_add_zero <= PCVRP.rubble_graph_size) & (medic_tour_add_zero > 0)).nonzero()[:,1].view(-1, PCVRP.rubble_graph_size))
            # engineer_rubble_victim_time = torch.gather(engineer_cumulative_time, 1, ((engineer_to

            medic_rubble_victim_time = torch.gather(medic_node_number_order_time, 1, ((medic_tour_sorted <= PCVRP.RUBBLE_GRAPH_SIZE) & (medic_tour_sorted > 0)).nonzero()[:,1].view(-1, PCVRP.RUBBLE_GRAPH_SIZE))
            engineer_rubble_victim_time = torch.gather(engineer_node_number_order_time, 1, ((engineer_tour_sorted <= PCVRP.RUBBLE_GRAPH_SIZE) & (engineer_tour_sorted > 0)).nonzero()[:,1].view(-1, PCVRP.RUBBLE_GRAPH_SIZE))

            late_rubble_penalty = torch.where(medic_rubble_victim_time < engineer_rubble_victim_time, 1, 0).sum(1)

            late_rubble_penalty = late_rubble_penalty.to(device=p.device)

            # jjj_a = ((medic_tour_add_zero <= PCVRP.rubble_graph_size) & (medic_tour_add_zero > 0)).nonzero()[:,1].view(-1, PCVRP.rubble_graph_size)
            # jjj_b = ((engineer_tour_add_zero <= PCVRP.rubble_graph_size) & (engineer_tour_add_zero > 0)).nonzero()[:,1].view(-1, PCVRP.rubble_graph_size)
            # big_mask = []
            # for run_id, run in enumerate(medic_tour_add_zero):
            #     inner_mask = []
            #     for node_id, node in enumerate(run):
            #         if node in dataset['rubbles'][run_id]:
            #             inner_mask.append(True)indices
            #         else:
            #             inner_mask.append(False)
            #     big_mask.append(inner_mask)
            #
            # valid_victim = torch.tensor(big_mask).nonzero()
            # valid_indices = torch.zeros((medic_cumulative_time.size(0), PCVRP.rubble_graph_size))
            # vii_col = 0
            # for vii in valid_victim:
            #     valid_indices[vii[0]][vii_col] = vii[1]
            #     vii_col += 1
            # medic_rubble_victim_time = torch.gather(medic_cumulative_time, 1, valid_indices)


            # print(dataset)
            # 228.3554 56045.7891
            # print(length.sum(0), high_value_mismatch_penalty.sum(0), late_rubble_penalty.sum(0))
            return length + high_value_mismatch_penalty * PCVRP.HIGH_VALUE_MISMATCH_PENALTY_COEFF + late_rubble_penalty * PCVRP.LATE_RUBBLE_PENALTY_COEFF, None

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
    
    def __init__(self, filename=None, size=28, num_samples=10000, offset=0, distribution=None):
        assert PCVRP.PLAYER_ROLE in ["medic", "engineer"]

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

            print("### Collecting data for PCVRP problem: ###")
            for i in tqdm(range(num_samples)):
                # t = torch.ones(size) * 5/55

                if PCVRP.PLAYER_ROLE == "medic":
                    prob = torch.ones(size) * 4 / 20
                    prize = torch.bernoulli(prob)
                    prize[prize == 0] = 0.1
                    prize[prize == 1] = PCVRP.HIGH_VALUE
                    self.data.append({
                        'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                        # Uniform 1 - 9, scaled by capacities
                        # 'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                        'demand': torch.ones(size) / PCVRP.MEDIC_TOOL_DURABILITY,
                        'depot': torch.FloatTensor(2).uniform_(0, 1),
                        'prize': prize
                    })
                elif PCVRP.PLAYER_ROLE == "engineer":
                    # medic_prob = torch.ones(medic_graph_size) * 4 / 20
                    # medic_prize = torch.bernoulli(medic_prob)
                    medic_prize = torch.ones(PCVRP.MEDIC_GRAPH_SIZE) * 0.1
                    medic_prize[-PCVRP.HIGH_VALUE_VICTIM_SIZE:] = PCVRP.HIGH_VALUE
                    victim_loc = torch.FloatTensor(PCVRP.MEDIC_GRAPH_SIZE, 2).uniform_(0, 1)
                    medic_batch = {
                        'loc': victim_loc.unsqueeze(0),
                        # Uniform 1 - 9, scaled by capacities
                        # 'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                        'demand': (torch.ones(PCVRP.MEDIC_GRAPH_SIZE) / PCVRP.MEDIC_TOOL_DURABILITY).unsqueeze(0),
                        'depot': torch.FloatTensor(2).uniform_(0, 1).unsqueeze(0),
                        'prize': medic_prize.unsqueeze(0)
                    }
                    # prize = torch.cat((torch.ones(medic_graph_size-high_value_victim_size)*0.1, torch.ones(high_value_victim_size)*high_value))

                    PCVRP.PLAYER_ROLE = 'medic'
                    PCVRP.MEDIC_MODEL.eval()
                    PCVRP.MEDIC_MODEL.set_decode_type('greedy')
                    with torch.no_grad():
                        length, log_p, pi = PCVRP.MEDIC_MODEL(medic_batch, return_pi=True)
                    medic_tour = pi
                    PCVRP.PLAYER_ROLE = 'engineer'


                    # rubble_sample = sample = torch.utils.data.WeightedRandomSampler(torch.arange(PCVRP.medic_graph_size), num_samples=PCVRP.rubble_graph_size, replacement=False)
                    # rubbles = torch.randint(0, PCVRP.medic_graph_size, (PCVRP.rubble_graph_size,))
                    # rubbles = torch.tensor([rs for rs in rubble_sample])
                    # rubbles = set()
                    # while len(rubbles) != PCVRP.rubble_graph_size:
                    #     rubbles.add(random.randint(0, PCVRP.med))

                    # selected_victim_loc = torch.gather(victim_loc, 0, torch.hstack((rubbles.view(-1, 1), rubbles.view(-1, 1))))
                    selected_victim_loc = victim_loc[:PCVRP.RUBBLE_GRAPH_SIZE]
                    high_value_loc = victim_loc[-PCVRP.HIGH_VALUE_VICTIM_SIZE:]
                    # print(len(selected_victim_loc), len(high_value_loc))
                    # print(torch.cat((selected_victim_loc, high_value_loc)).shape)
                    self.data.append({
                        'loc': torch.cat((selected_victim_loc, high_value_loc)),
                        # Uniform 1 - 9, scaled by capacities
                        # 'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                        'medic_tour': medic_tour.squeeze(0),
                        'victim_loc': victim_loc,
                        # 'rubbles': rubbles,
                        'demand': torch.ones(size) / PCVRP.ENGINEER_TOOL_DURABILITY,
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
