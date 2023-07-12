import face_recognition
import cv2
import numpy as np
import tkinter as tk
import tkinter.messagebox as mb
import os
from PIL import Image, ImageTk

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.
# これは、Webカメラからのライブビデオで顔認識を実行するデモです。それはより少し複雑です
# 他の例ですが、実行速度を大幅に向上させるための基本的なパフォーマンスの調整が含まれています。
#1  各ビデオフレームを1/4解像度で処理します（ただし、フル解像度で表示します）
#2  ビデオの1フレームおきに顔のみを検出します。



# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
#注意：この例では、Webカメラから読み取るためにのみOpenCV（ `cv2`ライブラリ）をインストールする必要があります。
#face_recognitionライブラリを使用するためにOpenCVは*必須*ではありません。これを実行する場合にのみ必要です
#特定のデモ。インストールに問題がある場合は、代わりにそれを必要としない他のデモを試してください。

#ウェブカメラ＃0（デフォルトのもの）への参照を取得します
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(1)


#サンプル画像をロードして、それを認識する方法を学びます。
# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]


#2番目のサンプル画像をロードし、それを認識する方法を学びます。
# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

#3番目のサンプル画像をロードし、それを認識する方法を学びます。
waka_image = face_recognition.load_image_file("waka.jpg")
waka_face_encoding = face_recognition.face_encodings(waka_image)[0]




#既知の顔エンコーディングとその名前の配列を作成する
# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    waka_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "waka waka"
]

#いくつかの変数を初期化します
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:

    #ビデオの単一フレームを取得します
    # Grab a single frame of video
    ret, frame = video_capture.read()

    #顔認識処理を高速化するために、ビデオのフレームのサイズを1 / 4サイズに変更します
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    #画像をBGRカラー（OpenCVが使用）からRGBカラー（face_recognitionが使用）に変換します
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]


    #時間を節約するために、ビデオの1フレームおきにのみ処理します
    # Only process every other frame of video to save time
    if process_this_frame:

        #ビデオの現在のフレーム内のすべての顔と顔のエンコーディングを検索します
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            #顔が既知の顔と一致するかどうかを確認します
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"


            # known_face_encodingsで一致が見つかった場合は、最初のものを使用してください。
            #一致する場合はTrue：
            #first_match_index = matches.index（True）
            #名前 = know_face_names[first_match_index]
            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            #または代わりに、新しい顔まで​​の距離が最小の既知の顔を使用します
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            print(name)

    process_this_frame = not process_this_frame



    #結果を表示する
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        #検出したフレームが1 / 4サイズに拡大縮小されたため、顔の位置を拡大縮小
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        #顔の周りにボックスを描く
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #顔の下に名前の付いたラベルを描く
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    #結果の画像を表示する
    # Display the resulting image
    cv2.imshow('Video', frame)




    #キーボードの「q」を押して終了します
    # Hit 'q' on the keyboard to quit!
#    if cv2.waitKey(1) & 0xFF == ord('q'):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(name)
        break




#ウェブカメラへのハンドルを解放します
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()