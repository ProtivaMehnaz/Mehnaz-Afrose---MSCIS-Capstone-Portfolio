# + Mehnaz Afrose

Table1 = [18.1, 15.4, 19.0, 13.4, 10.2,
          13.1, 18.1, 14.4, 15.0, 10.8, 5.4, 12.2]
Table2 = [0.7, 0.0, 0.7, 1.0, 1.1, 0.4, 0.0, 1.0, 2.3, 2.9, 1.3]


def Average(lst):
    sum_of_list = 0
    for i in range(len(lst)):
        sum_of_list += lst[i]
    average = sum_of_list/len(lst)
    return average


average1 = Average(Table1)
average2 = Average(Table2)
print(f'Average mortality rate before hand washing policy: {average1:.1f}')
print(f'Average mortality rate before hand washing policy: {average2:.1f}')
