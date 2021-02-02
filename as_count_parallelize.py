'''
make sure you have created the 10 folders to create the csv files
you can generate with:
mkdir stats_{0..9}
'''
import csv
with open("stats/as_count.csv", 'r', encoding="utf8") as as_count_file:
    reader = csv.reader(as_count_file)
    # set headers
    headers = next(reader, None)
    i = 0
    while i < 10:
        with open("stats_"+str(i%10)+"/as_count.csv", 'w', encoding="utf8") as as_count_file_out:
            writer = csv.writer(as_count_file_out)
            writer.writerow(headers)
        i = i + 1
    # split file in 10
    line_num = 0
    for line in reader:
        line_num = line_num + 1
        if line_num == 1901:
            break
        with open("stats_"+str(line_num%10)+"/as_count.csv", 'a', encoding="utf8") as as_count_file_out:
            writer = csv.writer(as_count_file_out)
            writer.writerow(line)