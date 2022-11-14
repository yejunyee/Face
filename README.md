# Face_Extraction_TeamProject

오늘은 딥러닝을 활용한 얼굴 추출 방법을 알아 볼 것이다.
알아 보기 전에 먼저 딥러닝에 대한 간단한 이해가 필요한데,
우리가 흔히 알고있는 인공지능이란 기술이 더 고차원적으로 발전한 기술이라 생각하면 된다.
이것을 정리해보자면 기술의 종류 중 하나인 인공신경망을 수많은 계층 형태로 연결한 기법으로 현대 인공지능의 기술로 인정받고 화제가 되고 있는 기술이다.
그렇다면 이것이 얼굴인식 기술과 어떻게 연결되어 사용될 수 있을까?

먼저 영상이나 이미지에서 데이터 즉 우리의 얼굴이 데이터값으로 들어가기 위해선 먼저 얼굴의 영역을 캐치해야 필요한 정보를 얻을 수 있는데,
여기서 컬러를 흑백으로 변환시키는 과정을 거친다.
그 이유는 컬러(rgb)는 각 3개의 채널로 보여지지만 흑백은 한 층의 영역에서만으로도 표현이 가능하기 때문이다.
그리고 그 영역의 명암 정보를 통해서 컴퓨터가 그 데이터를 통해 얼굴을 찾아주게 만들어준다.

여기서 궁금한건 얼굴 영역을 찾긴 하는데 아직 서로의 얼굴들을 판별하는데는 데이터가 부족한거 같은데 어떻게 하면 서로의 얼굴들을 구별할 수 있을까?
보통 우리가 사람들의 얼굴을 구별할 수 있는건 생김새 즉, 이모구비의 영향을 받는다는 것인데 이런 특별한 정보들을 통해 컴퓨터가 얼굴을 구분할 수 있도록 해줄 것이다.
앞에서 구한 데이터로 컴퓨터가 이목구비의 기준점을 찾아주고 딥러닝을 통해 보다 나은 기준점 개수나 얼굴을 더 잘 인식하도록 학습하는 과정을 반복하는데 이게 딥러닝 기술의 핵심이라 볼 수 있을 것 같다.

이제 얼굴을 구별할 어느 정도의 값들이 저장되고 정리됐다면 컴퓨터가 특정 이미지에 얼굴을 추출할 수 있게 된다.
이렇게 추출된 얼굴 정보는 임베딩이라는 과정을 거쳐 정보들을 어떠한 벡터로 실수화시켜준다.
실수화 된 데이터는 그 해당 이미지에 대한 사람의 이름과 함께 서버에 저장되면서 컴퓨터는 얼굴을 인식할 준비가 끝나간다.

이제 카메라로 얼굴을 보여주면 해당 영상에서 사람의 얼굴의 영역과 특징들을 찾고 임베딩 값을 구해 전에 저장해놨던 임베딩 값과 비교해 사람의 얼굴들을 구별하고 인식하게 된다.

얼굴인식은 현재 코로나로 인해 쓰임이 많아지고 다양해지는데 이러한 딥러닝 기술을 통해서 더욱 향상된 얼굴인식을 만들 수 있다는 사실에 놀라웠고 좀 더 깊이 배우고 싶은 학문이라 생각이 들었다.
딥러닝은 우리가 쓰고있는 순간에도 무섭게 발전하고 있는데,
사람이 더 살아가는데 도움이 되는 기술로 이끌 수 있도록 컴퓨터공학과 학생들이
열심히 공부해서 활용할 수 있도록 노력해야겠다고 생각했다.
출처:https://www.codingworldnews.com/news/articleView.html?idxno=3044
나무위키

# code

let video = document.getElementById('videoInput');
let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
let dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
let gray = new cv.Mat();
let cap = new cv.VideoCapture(video);
let faces = new cv.RectVector();
let classifier = new cv.CascadeClassifier();

// load pre-trained classifiers
classifier.load('haarcascade_frontalface_default.xml');

const FPS = 30;
function processVideo() {
    try {
        if (!streaming) {
            // clean and stop.
            src.delete();
            dst.delete();
            gray.delete();
            faces.delete();
            classifier.delete();
            return;
        }
        let begin = Date.now();
        // start processing.
        cap.read(src);
        src.copyTo(dst);
        cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
        // detect faces.
        classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
        // draw faces.
        for (let i = 0; i < faces.size(); ++i) {
            let face = faces.get(i);
            let point1 = new cv.Point(face.x, face.y);
            let point2 = new cv.Point(face.x + face.width, face.y + face.height);
            cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
        }
        cv.imshow('canvasOutput', dst);
        // schedule the next one.
        let delay = 1000/FPS - (Date.now() - begin);
        setTimeout(processVideo, delay);
    } catch (err) {
        utils.printError(err);
    }
};

// schedule the first one.
setTimeout(processVideo, 0);

# A letter of appreciation

이시원: 영상처리를 이용하여 얼굴인식을 이렇게 
할수 있다는 것에 놀랐으며, 이때까지 영상처리 수업에서 배웠던 것들
과 접목하여 또 다른 것을 제작하기위한 기반 기술이 늘어난것 같아서 뿌듯한
팀 과제가 되었던것 같습니다.

지영수: 영상처리를 이용한 얼굴인식 팀프로젝트를
계획해보니 영상처리의 응용이 이렇게나 다양하구나 느낄 수 있었습니다.
계획한 팀프로젝트를 완성할 수 있도록 더 많은 공부와 연습을 하겠다고 다짐했습니다.

이예준: 평소에 막연히 그냥 얼굴인식은 영상처리 기술 중 하나이다라고만
생각했는데 이렇게 팀 프로젝트를 통해서 자세하게 알아보니 딥러닝을 통해서 더 고차원적인
얼굴인식 기술을 구현할 수 있다는게 놀라웠습니다.
저희가 이번에 구현한 얼굴인식은 비록 opencv를 활용해 간단한 얼굴영역만 보여줬지만
다가올 팀플에서는 좀 더 연구해서 성능들을 추가해 저희만의 얼굴인식 기술을 만드는 것을
목표로 하고 열심히 공부해야겠다는 생각이 들었습니다.

# Simulation
https://user-images.githubusercontent.com/62829568/201556231-83456de1-5dae-4f9e-a7d4-15284895a30f.mp4

