def ask(question):
    while True:
        i = input(question + " [y/n] ")
        if i and i in 'yY':
            return True
        if i and i in 'nN':
            return False
