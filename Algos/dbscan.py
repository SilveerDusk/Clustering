import sys

def main():
  if len(sys.argv) != 4:
    print("Usage: python3 dbscan <Filename> <epsilon> <NumPoints>")
    sys.exit(1)

  datafile = sys.argv[1]
  epsilon = sys.argv[2]
  numPoints = sys.argv[3]


  pass




if __name__ == "__main__":
  main()