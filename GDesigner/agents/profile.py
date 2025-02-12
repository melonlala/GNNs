import random
def get_profile(N,profile_choice):
        profile_pool =['Math Solver', 'Mathematical Analyst', 'Programming Expert', 'Inspector']
        # if len(profile_pool) >= N:
        #     return random.sample(profile_pool, N)
        # else:
        #     return profile_pool + random.sample(profile_pool, N-len(profile_pool))
        
        # profiles = [random.sample(profile_pool, 1)[0] for _ in range(N)]
        profile_dict = {
            3: [
                    ['Math Solver', 'Mathematical Analyst', 'Programming Expert'],
                     ['Math Solver', 'Mathematical Analyst', 'Inspector'],
                     ['Math Solver', 'Programming Expert', 'Inspector'],
                     ['Mathematical Analyst', 'Programming Expert', 'Inspector']
                ],
            4: [
            ['Math Solver', 'Mathematical Analyst', 'Programming Expert', 'Inspector']
            ],
            5: [
            ['Math Solver', 'Mathematical Analyst', 'Programming Expert', 'Inspector', 'Math Solver'],
            ['Math Solver', 'Mathematical Analyst', 'Programming Expert', 'Inspector', 'Mathematical Analyst'],
            ['Math Solver', 'Mathematical Analyst', 'Programming Expert', 'Inspector', 'Programming Expert'],
            ['Math Solver', 'Mathematical Analyst', 'Programming Expert', 'Inspector', 'Inspector']
            ],

            }

        if N > 5:  
            profiles = profile_pool + random.sample(profile_pool, N-len(profile_pool))
        else:
            profiles = profile_dict[N][profile_choice]
        return profiles
        # return profiles