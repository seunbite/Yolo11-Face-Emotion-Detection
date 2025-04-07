from ultralytics import YOLO
import cv2
import imageio
import numpy as np
import os
import fire
from matplotlib import pyplot as plt
import matplotlib.collections as mcoll

# 모델 로드
model = YOLO('best.onnx')
cmap = plt.get_cmap('tab10')
color_list = ['white'] + [cmap(i) for i in range(7)]  # 7개 감정에 대해 7가지 색상
emotions_list = ['no face', 'neutral', 'happy', 'sad', 'angry', 'surprise', 'disgust', 'fear']


def main(
    gif_path: str = 'putin.gif',
):
    output_dir = gif_path.replace('.gif', '_output')
    os.makedirs(output_dir, exist_ok=True)
    frame_timestamps = []
    frame_emotions = []
    segments = []
    seg_colors = []

    # GIF의 모든 프레임을 불러오기
    frames = imageio.mimread(gif_path)

    for frame_idx, frame in enumerate(frames):
        # 프레임 형식 변환 (RGBA/RGB -> BGR)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 그레이스케일 변환 후 3채널로 병합
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_image_3d = cv2.merge([gray_image, gray_image, gray_image])
        
        # 모델 추론 실행
        results = model(gray_image_3d)
        result = results[0]
        
        # 결과 시각화
        try:
            annotated_frame = result.plot()
        except AttributeError:
            print(f"Error: plot() method not available for results in frame {frame_idx}.")
            continue
       
        if not result.boxes:
            print(f"No faces detected in frame {frame_idx}.")
            class_id = 0
            class_prob = 0.0
            class_name = "No Face"
        
        else:
           for box_idx, box in enumerate(result.boxes):
                # 가장 확률이 높은 클래스 찾기
                class_id = int(box.cls)
                class_prob = box.conf.item()
                class_name = result.names[class_id]
                
                # 로짓 값 (확률) 출력
                print(f"Frame {frame_idx}, Face {box_idx}: Emotion={class_name}, Confidence={class_prob:.4f}")
                
                # 결과 이미지에 정보 추가
                info_text = f"{class_name}: {class_prob:.2f}"
                cv2.putText(annotated_frame, info_text, 
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1])-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 결과 이미지 저장
        frame_timestamps.append(frame_idx / 10)  # 예시로 10fps 가정
        frame_emotions.append(class_id)
        
        output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(output_path, annotated_frame)
        print(f"Saved: {output_path}")

    print("All frames processed and saved.")
    
    for i in range(len(frame_timestamps) - 1):
        segments.append([(frame_timestamps[i], frame_emotions[i]),
                        (frame_timestamps[i+1], frame_emotions[i+1])])
        seg_colors.append(color_list[frame_emotions[i]])

    lc = mcoll.LineCollection(segments, colors=seg_colors, linewidth=2)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.add_collection(lc)
    ax.set_xlim(frame_timestamps[0], frame_timestamps[-1])
    ax.set_ylim(-0.5, 6.5)
    ax.set_yticks(range(7))
    ax.set_yticklabels(emotions_list)
    ax.set_xlabel("Time (s)")
    ax.set_title("Time-varying Emotion Classification (GIF)")
    plt.savefig(os.path.join(output_dir, "emotion_plot.png"))


if __name__ == "__main__":
    fire.Fire(main)