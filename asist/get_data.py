import pickle             

depot = [0, 0]
prize = [1, 1, 1, 1, 1]
max_length = 5.
loc = [
    [0.3, 0.1],
    [0.1, 0.3],
    [0.3, 0.3],
    [0.2, 0.4],
    [0.5, 0.5]
]


some_obj = [(depot, loc, prize, max_length)]



with open('falcon_tmp.pkl', 'wb') as f:
    pickle.dump(some_obj, f)


with open('falcon_tmp.pkl', 'rb') as f:
    loaded_obj = pickle.load(f)

print('loaded_obj is', loaded_obj)