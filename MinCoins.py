import time

def timefunction(func):

    def wrapper(*args):
        start = time.time()
        func(*args)
        return time.time() - start

    return wrapper

@timefunction
def mincoinfinder(coins = [1],cents = 0):
    #Given a set of coin values returns the minimum number of coins needed to add up to the cents value
    #We assume that the coin worth 1 cent is used
    #Strategy is using the classical dynamic programming approach (proof by induction)
    #Added coin tracker table to determine the coins needed to make the change
    mincoins = [[0 for j in range(cents+1)] for i in range(len(coins))]
    mincoins[0] = [j for j in range(cents+1)]

    coin_tracker = {coins[x]:0 for x in range(len(coins))}
    coin_tracker_table = [[coin_tracker.copy() for j in range(cents+1)] for i in range(len(coins))]
    for i in range(len(coin_tracker_table[0])):
        coin_tracker_table[0][i][1] = i

    for i in range(1,len(coins)):
        for j in range(cents+1):
            if coins[i] > j:
                mincoins[i][j] = mincoins[i-1][j]
                coin_tracker_table[i][j] = coin_tracker_table[i-1][j].copy()
            else:
                if mincoins[i-1][j] < mincoins[i][j-coins[i]]+1:
                    mincoins[i][j] = mincoins[i-1][j]
                    coin_tracker_table[i][j] = coin_tracker_table[i-1][j].copy()

                else:
                    mincoins[i][j] = mincoins[i][j-coins[i]]+1
                    coin_tracker_table[i][j] = coin_tracker_table[i][j-coins[i]].copy()
                    coin_tracker_table[i][j][coins[i]] += 1
               # mincoins[i][j] = min(mincoins[i-1][j],mincoins[i][j-coins[i]]+1)
    return mincoins[-1][-1]


coins = [1,10,25]

print mincoinfinder(coins,31)
