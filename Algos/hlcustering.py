import sys

def main():
  if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Usage: python3 hlclustering <Filename> [<threshold>]")
    sys.exit(1)

  datafile = sys.argv[1]
  if len(sys.argv) == 3:
    threshold = sys.argv[2]


  pass




if __name__ == "__main__":
  main()