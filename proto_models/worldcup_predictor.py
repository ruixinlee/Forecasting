import random
from collections import Counter
countries = ['France', 'Uruguay', 'Brazil', 'Belgium', 'Croatia', 'Russia', 'England', 'Sweden']

n_list = [i for i in range(100,500)]
nth_draw = random.choice(n_list)

draws = []
for i in range(1,nth_draw):
    draws.append(random.choice(countries))

print(Counter(draws))


