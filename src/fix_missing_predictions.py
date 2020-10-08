import os
import pickle

missing_frame_list = []
with open('missing_frame_list.txt', 'rb') as fp:
    missing_frame_list.extend(pickle.load(fp))

os.chdir("../results")
prediction_text_file_list = os.listdir()

corrected_frame_list = []
for i in range(len(prediction_text_file_list)):
    _list = []
    with open(prediction_text_file_list[i], "r") as fd:
        for line in fd:
            line = line.strip()
            _list.append(line)

        for j in range(len(missing_frame_list[i])):
            _list.insert(missing_frame_list[i][j], "-1")  # skip header

        _list.remove("Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise")

    corrected_frame_list.append(_list)

if not os.path.exists('../corrected'):
    os.makedirs('../corrected')

os.chdir('../corrected')

for i in range(len(corrected_frame_list)):
    f = open(str(prediction_text_file_list[i]), "w+")
    for j in range(len(corrected_frame_list[i])):
        f.write(corrected_frame_list[i][j] + "\n")
    f.close()

os.chdir('../src')
filename_expected_frame_list = []
with open("Expr_Challenge_video_and_total_number_of_frames.txt", "r") as fd:
    for line in fd:
        line = line.strip()
        line = line.split()
        filename = line[2] + ".txt"
        frame_no = int(line[9])
        _list = [filename, frame_no]
        filename_expected_frame_list.append(_list)

os.chdir('../corrected')
files = os.listdir()

filename_current_frame_list = []
for i in range(len(files)):
    count = 0
    with open(files[i], "r") as fd:
        for line in fd:
            count += 1

        filename_current_frame_list.append(count)

for i in range(len(files)):
    f = open(str(files[i]), "a")
    for j in range(len(filename_expected_frame_list)):
        if files[i] == filename_expected_frame_list[j][0]:
            frame_no = filename_expected_frame_list[j][1]
            for k in range(frame_no - filename_current_frame_list[i]):
                f.write("-1" + "\n")
    f.close()

for i in range(len(files)):
    f = open(str(files[i]), "r")
    header_line = "Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise" + "\n"
    lines = f.readlines()
    lines.insert(0, header_line)
    f.close()

    # We again open the file in WRITE mode
    f = open(str(files[i]), "w")
    f.writelines(lines)
    f.close()

filename_output_frame_list = []
for i in range(len(files)):
    count = 0
    with open(files[i], "r") as fd:
        for line in fd:
            count += 1

        _list = [files[i], count - 1] # disregard header line
    filename_output_frame_list.append(_list)

sorted_filename_output_frame_list = sorted(filename_output_frame_list)
sorted_filename_expected_frame_list = sorted(filename_expected_frame_list)

print(sorted_filename_output_frame_list)
print(sorted_filename_expected_frame_list)

print(sorted_filename_output_frame_list == sorted_filename_expected_frame_list)
