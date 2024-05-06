# Mehnaz Afrose

Guests_List = []
for i in range(0, 10000):
    x = input("Enter a guest's name or type 'end' to stop.")
    y = x.upper()
    if y == "END":
        break
    else:
        Guests_List.append(x)
for Guest in Guests_List:
    print(Guest)
print(
    f'You have invited {len(Guests_List)} guests at a cost of {len(Guests_List)*12} for food.')
