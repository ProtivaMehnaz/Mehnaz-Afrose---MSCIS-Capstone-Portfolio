# Mehnaz Afrose
# Determines the shipping cost based on the ship_to_state.
# When testing, change the ship_to_state to various state abreviations
# to verify your code works for all conditions.
ship_to_state = input(
    "Enter a state to ship to, e.g. TX, NM, OK, NY, AK, HI, etc.: ")
ship_to_state = ship_to_state.upper()
shipping_cost = 0
if ship_to_state == "TX" or ship_to_state == "NM" or ship_to_state == "OK":
    print(f"Shipping to {ship_to_state} costs {shipping_cost}")
elif ship_to_state == "NY":
    print(f"Shipping to {ship_to_state} costs 7")
elif ship_to_state == "AK":
    print(f"Shipping to {ship_to_state} costs 15.75")
elif ship_to_state == "HI":
    print(f"Shipping to {ship_to_state} costs 20")
else:
    print(f"Shipping to {ship_to_state} costs 12.5")
