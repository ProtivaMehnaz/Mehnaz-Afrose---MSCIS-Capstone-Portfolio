import requests
"""
Documentation at 
https://documenter.getpostman.com/view/10808728/SzS8rjbc
"""

# 1. get the API data from the web server
# This API server does not require an API key
# 2. Convert the data into a useful python objects: usually a list or dictionaries
url = "https://api.covid19api.com/summary"
response = requests.get(url)
data = response.json()
print(type(data))

global_stats = data['Global']
death = global_stats['TotalDeaths']

confirmed = global_stats['TotalConfirmed']

mortality_rate = death/confirmed

print(f'Total Deaths: {death}')
print(f'Total Confirmed Cases: {confirmed}')
print(f'Mortality Rate: {mortality_rate}')
