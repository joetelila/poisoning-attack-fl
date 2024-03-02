class Modelutils:
    def __init__(self, name):
        self.name = name

    def attack(self):
        print(f"Hello, {self.name}!")

    def defend(self, num1, num2):
        return num1 + num2