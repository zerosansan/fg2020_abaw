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
    f.write("Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise\n")
    for j in range(len(corrected_frame_list[i])):
        f.write(corrected_frame_list[i][j] + "\n")
    f.close()
