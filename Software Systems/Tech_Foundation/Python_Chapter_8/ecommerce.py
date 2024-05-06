# Do not modify this file. Its used to example how to import modules and functions

def calc_tax(amount):
    return amount * .085

def calc_shipping(state): 
    if state.lower == "tx":
        return 0
    else: 
        return 10.25


if __name__ == "__main__":
    # test code runs when ecommerce.py is run directly from the terminal
    print(calc_tax(100))
    print(calc_shipping("TX"))