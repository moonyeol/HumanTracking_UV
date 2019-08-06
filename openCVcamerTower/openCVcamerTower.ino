//OpenCV 카메라 타워 코드

import processing.serial.*;

// Combining GSVideo capture with the OpenCV library for face detection
// http://ubaa.net/shared/processing/opencv/
import hypermedia.video.*;
import java.awt.Rectangle;
import codeanticode.gsvideo.*;

OpenCV opencv;
GSCapture cam;

// 대비, 밝기 조절 변수
int contrast_value    = 0;
int brightness_value  = 0;

//수직, 수평으로 움직이는 서보모터를 구별하기 위한 변수
char verticalSignal = 0;
char horizonSignal = 1;

//서보모터 각도의 초기값 지정
char servoHPosition = 90;
char servoVPosition = 90;

//얼굴 중앙값 초기화
int midFaceY=0;
int midFaceX=0;

//화면의 중심좌표값 지정
int midScreenY = (480/2);
int midScreenX = (640/2);
int midScreenWindow = 10; //화면 중앙에서 어느정도 위치안에 얼굴의 중앙위치점가 들어올 경우 
                          //스크린에서 중앙으로 들어왔다고 인식할 것인지 오차범위 지정

int stepSize=1; //모터 이동값 지정

Serial port;


void setup() {
  size(640, 480);
    
  cam = new GSCapture(this, 640, 480);
  cam.start();
  //OpenCV사용 초기화
  opencv = new OpenCV(this);

  opencv.allocate(640,480);   
  
  // "haarcascade_frontalface_alt.xml"을 불러와서 얼굴의 앞을 인식한다
  opencv.cascade( OpenCV.CASCADE_FRONTALFACE_ALT );  
  
  println(Serial.list()); 
  println(midScreenX);
  println(midScreenY);
  
  //시리얼통신을 위한 포트 생성
  port = new Serial(this, Serial.list()[0], 57600); 

  //메시지 프린트
  println("Drag mouse on X-axis inside this sketch window to change contrast");
  println("Drag mouse on Y-axis inside this sketch window to change brightness");
  
  //서보모터 초기각도 전송
  port.write(horizonSignal);   
  port.write(servoHPosition);  
  port.write(verticalSignal);       
  port.write(servoVPosition);   
  
}

void captureEvent(GSCapture c) {
  c.read();
}

public void stop() {
  opencv.stop();
  super.stop();
}

void draw() {
  opencv.copy(cam);
    
  opencv.convert(GRAY);
  opencv.contrast(contrast_value);
  opencv.brightness(brightness_value);

  //OpenCV라이브러리를 이용하여 사람의 얼굴 앞면을 인식한다.
  Rectangle[] faces = opencv.detect(1.2, 2, OpenCV.HAAR_DO_CANNY_PRUNING, 40, 40);

  //캠에서 인식하는 이미지를 화면에 출력하여 영상으로 만든다
  image(cam, 0, 0);

  //인식한 얼굴의 테두리에 사각형을 그린다.
  noFill();
  stroke(0, 255, 0);
  strokeWeight(5);
  for(int i = 0; i < faces.length; i++) {
    rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height); 
  }
  
  //얼굴의 길이가 0보다 클 경우(사람의 얼굴을 인식하고 있을 경우)
  if(faces.length > 0){
    //현재 인식하고 있는 얼굴의 중앙 점을 구하여 midFaceX와 midFaceY에 저장
    midFaceX = faces[0].x + (faces[0].width/2); 
    midFaceY = faces[0].y + (faces[0].height/2);
    
    //현재 얼굴의 위치가 스크린의 중앙보다 아래에 위치할 경우 수직으로 움직이는 서보모터의 각도를 1도씩 감소시킨다
    if(midFaceY < (midScreenY - midScreenWindow)){
      if(servoVPosition >= 5)
        servoVPosition -= stepSize;
    }

    //현재 얼굴의 위치가 스크린의 중앙보다 위에 위치할 경우 수직으로 움직이는 서보모터의 각도를 1도씩 증가시킨다
    else if(midFaceY > (midScreenY + midScreenWindow)){
      if(servoVPosition <= 175)
        servoVPosition +=stepSize; 
    }

    //현재 얼굴의 위치가 스크린의 중앙보다 왼쪽에 위치할 경우 수평으로 움직이는 서보모터의 각도를 1도씩 감소시킨다
    if(midFaceX < (midScreenX - midScreenWindow)){
      if(servoHPosition >= 5)
        servoHPosition -= stepSize; 
    }

    //현재 얼굴의 위치가 스크린의 중앙보다 아래에 위치할 경우 수평으로 움직이는 서보모터의 각도를 1도씩 증가시킨다
    else if(midFaceX > midScreenX + midScreenWindow){
      if(servoHPosition <= 175)
        servoHPosition +=stepSize; 
    }
    
  }
  
  //각자 서보모터의 각도를 시리얼통신을 통해 전송
  port.write(horizonSignal);      
  port.write(servoHPosition); 
  port.write(verticalSignal);        
  port.write(servoVPosition);  
  delay(1);
  
}

//밝기 대조값 변경
void mouseDragged() {
  contrast_value   = int(map(mouseX, 0, width, -128, 128));
  brightness_value = int(map(mouseY, 0, width, -128, 128));
}
