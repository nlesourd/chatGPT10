def main():
    keywords = ["exit()", "help()", "clear()"]
    print("Welcome to ChatGPT10")
    end = False
    while not(end):
        query = input('>> Enter your query : ')
        if query == "exit()":
            end = True
main()