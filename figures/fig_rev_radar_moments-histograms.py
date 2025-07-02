



def main():
    
    
    # read radar data
    from readers.radar import read_radar_multiple
    ds = read_radar_multiple()
    
    print(ds)
    # read radar reflectivity
    
    
    
if __name__ == "__main__":
    main()