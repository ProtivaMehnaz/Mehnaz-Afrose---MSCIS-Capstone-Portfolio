# MEHNAZ AFROSE
# Follow the instructions for coding a weather app
from statistics import mean

Daily_HT_LT_HU = [[62, 64, 79, 52, 46, 50, 58, 66, 71, 75, 78, 78, 76, 80, 77],
                  [42, 48, 47, 26, 28, 28, 32, 37, 43, 46, 48, 47, 48, 49, 50, ],
                  [0.48, 0.53, 0.46, 0.44, 0.4, 0.6, 0.58, 0.5, 0.48, 0.43, 0.41, 0.39, 0.39, 0.3, 0.4]]

Max_High_Temp = max(Daily_HT_LT_HU[0])
Average_High_Temp = mean(Daily_HT_LT_HU[0])
Min_Low_Temp = min(Daily_HT_LT_HU[1])
Average_Low_Temp = mean(Daily_HT_LT_HU[1])
Average_Humidity = mean(Daily_HT_LT_HU[2])

print(f'Weather forecast for the next {len(Daily_HT_LT_HU[0])} days: The average low will be {round(Average_Low_Temp, 0):.0f} and average high will be {round(Average_High_Temp, 0):.0f}. The average humidity will be {round(Average_Humidity, 2)}. The highest temperature will be {round(Max_High_Temp, 0):.0f}. The lowest temperature will be {round(Min_Low_Temp, 0):.0f}.')
