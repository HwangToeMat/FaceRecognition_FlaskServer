# FaceRecognition_FlaskServer  <a href="https://hwangtoemat.github.io/ai-project/2020-07-01-%EC%96%BC%EA%B5%B4%EC%9D%B8%EC%8B%9D%EA%B8%B0%EB%B0%98-%EC%8B%A4%EC%8B%9C%EA%B0%84-%EA%B5%90%EC%9C%A1-%ED%94%8C%EB%9E%AB%ED%8F%BC-%EA%B0%9C%EB%B0%9C/">[detail]</a>

얼굴인식 시스템의 파이프라인을 Flask를 통해 서버에서 구현한다.

## Demo

<img src="https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/AI-Project/image/KHUFACE/3/video.gif?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

## Usage

```
usage: model_master.py [-h] [--miniface MINIFACE] [--update UPDATE]

optional arguments:
  -h, --help           show this help message and exit
  --miniface MINIFACE  Minimum face to be detected. derease to increase accuracy. Increase to increase speed
  --update UPDATE      whether perform update the facebank
```

## 얼굴인식 모델의 파이프라인

얼굴인식 모델의 파이프라인은 크게 **두 단계**로 나누어진다.

**1. MTCNN을 활용한 얼굴위치 확인** <a href="https://hwangtoemat.github.io/paper-review/2020-03-28-MTCNN-%EB%82%B4%EC%9A%A9/">[MTCNN에 대한 상세설명]</a>

<img src="https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/AI-Project/image/KHUFACE/2/img1.png?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

  MTCNN을 통해 들어온 영상에서 얼굴의 위치를 검출한다. MTCNN은 그림과 같이 세 가지 네트워크를 통해 영상에 있는 여러 얼굴의 위치를 빠르게 파악할 수 있기때문에 본 프로젝트에 적합하여 이를 채택하였다.

**2. FaceNet을 활용한 얼굴의 특징 추출 및 신원 확인** <a href="https://hwangtoemat.github.io/paper-review/2020-04-02-FaceNet-%EB%82%B4%EC%9A%A9/">[FaceNet에 대한 상세설명]</a>

<img src="https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/AI-Project/image/KHUFACE/2/img2.png?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

  FaceNet을 통해 앞서 검출된 얼굴에서 128개의 특징값을 추출한다. 이 값들을 다시 128차원으로 임베딩하면 각 얼굴(사람)간의 유클리드 거리를 계산하여 얼굴을 구분할 수 있게 된다.
  
## 직접 실험한 영상

<img src="https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/AI-Project/image/KHUFACE/2/video.gif?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

## 딥러닝 서버와 웹 서버간의 통신

딥러닝 서버와 웹 서버는 Rest API방식으로 통신한다. (설계한 딥러닝 서버와 통신하는 웹 서버는 **유정수 엔지니어님** <a href="https://youjeongsue.tistory.com/">[유정수님 티스토리 링크]</a> 이 설계해 주셨다.)
그 중 GET을 사용하여 영상을 제공받는 카메라의 IP를 웹으로 부터 받고, 그 영상을 분석하여 얻은 정보를 웹으로 다시 보내주는 방식을 사용한다. 이때 사용되는 3가지 함수는 아래와 같다.

### 얼굴위치 확인 함수 - Modelfr

- Code

```python
def get_face(URL, device, targets=target, names=name):
    student_list = []
    frame = URL2Frame(URL)
    try:
        bboxes, landmarks = MTCNN_NET(frame, 0.5, device, 'MTCNN/weights/pnet_Weights',
                                      'MTCNN/weights/rnet_Weights', 'MTCNN/weights/onet_Weights')
        landmarks = landmarks.tolist()
        for i, b in enumerate(bboxes):
            box = dict()
            box['x1'] = b[0]
            box['y1'] = b[1]
            box['x2'] = b[2]
            box['y2'] = b[3]
            box['landmarks'] = landmarks[i]
            student_list.append(box)
        return student_list
    except:
        return []
```

- input

```
ip = http://xxx.xxx.xxx.xxx:xxxx/shot.jpg # ip camera의 영상 주소
landmark = [xxxx, xxxx, xxxx.....] # modelfr에서 찾은 얼굴의 landmark
```

- ouput

```
{"box":[{"landmarks":[xxxx, xxxx, xxxx.....],"x1":xxx,"x2":xxx,"y1":xxx,"y2":xxx},
        {"landmarks":[yyyy, yyyy, yyyy.....],"x1":yyy,"x2":yyy,"y1":yyy,"y2":yyy}]} 
        # 인식한 학생의 얼굴에서 찾은 landmark와 bounding box의 
```

- 실행화면

<img src="https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/AI-Project/image/KHUFACE/3/img1.png?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

**Modelfr은 수업시간 중 웹에서 계속해서 요청을 하는 함수이며 얼굴의 위치를 파악하여 웹으로 실시간 전송한다. 그리고 웹에서는 이러한 정보를 바탕으로 얼굴의 위치에 맞는 버튼을 실시간으로 생성힌다.**

### 얼굴 정보 확인 함수 - Modelin

- Code

```python
def get_id(URL, device, targets=target, names=name):
    student_list = []
    frame = URL2Frame(URL)
    try:
        faces = Face_alignment(
            frame, default_square=True, landmarks=landmarks)

        embs = []

        test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        for img in faces:
            embs.append(detect_model(
                test_transform(img).to(device).unsqueeze(0)))

        if embs != []:
            source_embs = torch.cat(embs)
            diff = source_embs.unsqueeze(-1) - \
                targets.transpose(1, 0).unsqueeze(0)
            dist = torch.sum(torch.pow(diff, 2), dim=1)
            minimum, min_idx = torch.min(dist, dim=1)
            min_idx[minimum > ((75-156)/(-80))] = -1
            results = min_idx

            for i, k in enumerate(bboxes):
                if results[i] == -1:
                    continue
                student_list.append(names[results[i] + 1])
        return student_list
    except:
        return []
```

- input

```
ip = http://xxx.xxx.xxx.xxx:xxxx/shot.jpg # ip camera의 영상 주소
landmark = [xxxx, xxxx, xxxx.....] # modelfr에서 찾은 얼굴의 landmark
```

- ouput

```
{"id":["0"]} # 인식한 학생의 id 값
```

- 실행화면

<img src="https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/AI-Project/image/KHUFACE/3/img2.png?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

**Modelin은 웹에 생성된 버튼을 클릭할 시 해당 얼굴의 특징과 DB의 정보를 비교하여 신원을 확인하여 웹으로 전송한다. 이를 통해 웹에서는 해당 인원에 대한 정보를 가져와 사용자가 쉽게 확인 및 수정 할 수 있도록 화면을 제공한다.**

### 출결 정보 확인 함수 - Modelat

- Code

```python
def get_id(URL, device, targets=target, names=name):
    student_list = []
    frame = URL2Frame(URL)
    try:
        bboxes, landmarks = MTCNN_NET(frame, 0.5, device, 'MTCNN/weights/pnet_Weights',
                                  'MTCNN/weights/rnet_Weights', 'MTCNN/weights/onet_Weights')
        faces = Face_alignment(
            frame, default_square=True, landmarks=landmarks)

        embs = []

        test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        for img in faces:
            embs.append(detect_model(
                test_transform(img).to(device).unsqueeze(0)))

        if embs != []:
            source_embs = torch.cat(embs)
            diff = source_embs.unsqueeze(-1) - \
                targets.transpose(1, 0).unsqueeze(0)
            dist = torch.sum(torch.pow(diff, 2), dim=1)
            minimum, min_idx = torch.min(dist, dim=1)
            min_idx[minimum > ((75-156)/(-80))] = -1
            results = min_idx

            for i, k in enumerate(bboxes):
                if results[i] == -1:
                    continue
                student_list.append(names[results[i] + 1])
        return student_list
    except:
        return []
```

- input

```
ip = http://xxx.xxx.xxx.xxx:xxxx/shot.jpg # ip camera의 영상 주소
```

- ouput

```
{"id":["4","2","1","0"]} # 인식한 학생의 id 값
```

- 실행화면

<img src="https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/AI-Project/image/KHUFACE/3/img3.png?raw=true" style="max-width:100%;margin-left: auto; margin-right: auto; display: block;">

**Modelat는 웹에서 출석확인 버튼을 클릭할 시 화면에서 보이는 모든 학생들의 정보를 웹으로 전송한다. 이를 통해 웹에서는 출석정보를 업데이트하여 사용자가 쉽게 확인할 수 있도록 화면을 제공한다.**
