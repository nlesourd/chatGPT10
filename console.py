import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
import pyterrier as pt
import preprocessing as ppc
import method

if not pt.started():
    pt.init()

## Open the inverted index
INVERTED_INDEX_PATH = "./data/inverted_index"
inverted_index = ppc.load_inverted_index_trec(INVERTED_INDEX_PATH)

## Define methods
baseline = method.Baseline(inverted_index)
advanced_method = method.AdvancedMethod(inverted_index)

def main():
    keywords_exp = {"help()" : "print the available commands", 
                "exit()": "to exit the program", 
                "clear()" : "to clear the shell",
                }

    print("Welcome to ChatGPT10")
    end = False
    while not(end):
        query = input('>> Enter your query : ')
        if query == "exit()":
            end = True
        elif query == "clear()":
            if os.name == 'posix':  # For Unix/Linux/Mac systems
                os.system('clear')
            elif os.name == 'nt':   # For windows systems
                os.system('cls')
        elif query == "help()":
            for keyword in keywords_exp:
                print("   - " + keyword + " --> " + keywords_exp[keyword])
        else:
            # Find the 1000 best results
            results = advanced_method.rank_query(query)
            print(results)

# Run the main
main()