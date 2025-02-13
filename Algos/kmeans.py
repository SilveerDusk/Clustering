import sys

def main():
  if len(sys.argv) != 3:
    print("Usage: python3 dbscan <Filename> <k>")
    sys.exit(1)

  datafile = sys.argv[1]
  k = sys.argv[2]


  pass




if __name__ == "__main__":
  main()