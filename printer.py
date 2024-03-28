import detectorlib

path2='2021_All raw data_No_background.xlsx'
graph=detectorlib.printer()
graph.load(path2, 4)

graph.print_all()