//속도 파트

int spd = 80; // 전체 PWM speed (0~255)

int frontLeftspd = 48;
int frontRightspd = 50;
int backLeftspd = 68; // 모터가 문제가 많다
int backRightspd = 57;

int clispd = 100; // 언덕일때 

int moveBackFrontLeftspd = 52; 
int moveBackFrontRightspd = 48;
int moveBackBackLeftspd = 55;
int moveBackBackRightspd = 50;

int leftmovefrontleftspd = 60;
int leftmovefrontrightspd = 58;
int leftmovebackleftspd = 78;
int leftmovebackrightspd = 56;

int Leftmovefrontleftspd = 62;
int Leftmovefrontrightspd = 56;
int Leftmovebackleftspd = 0;
int Leftmovebackrightspd = 60;

int rightmovefrontleftspd = 68;
int rightmovefrontrightspd = 64;
int rightmovebackleftspd = 82;
int rightmovebackrightspd = 60;

int Rightmovefrontleftspd = 60;
int Rightmovefrontrightspd = 56;
int Rightmovebackleftspd = 78;
int Rightmovebackrightspd = 0;

//속도 파트end

//br값이 강하게 나와서 br b값만 낮추기 ㅇㅇ

void setup() 
{
  Serial.begin(115200);
}
//왼쪽 오른쪽 돌때는 좀더 강한 pwm을 줘야함
void loop()
{
  if(Serial.available())
  {
    char a;
    a = Serial.read();
    switch(a)
    {
      case 'g':
      Serial.println("Forward");
      moveForward();
      break;

      case 'b':
      Serial.println("Backward");
      moveBackward();
      break;

      case 'r':
      Serial.println("moveRight");
      moveRight();
      break;

      case 'l':
      Serial.println("moveLeft");
      moveLeft();
      break;

      case 's':
      Serial.println("stopp");
      stopp();
      break;

      case 'S': // 움직임 도중에 멈추는것 관성 생각 ㅇㅇ (뒤로 좀 주고 멈추기)
      Serial.println("movestopp");
      movestopp();
      break;

      case 'R':
      Serial.println("turnRight");
      turnRight();
      break;

      case 'L':
      Serial.println("turnLeft");
      turnLeft();
      break;

      case 'G': //오르막길
      Serial.println("moveForwardCli");
      moveForwardCli();
      break;

      case 'B': //오르막길
      Serial.println("moveBackwardCli");
      moveBackwardCli();
      break;

      default:
        Serial.println("Wrong Serial");
        Serial.println(a);

    }
  }
}

void moveForward() 
{
  Serial.println("motorForward.");
  digitalWrite(44, HIGH);      // BR
  digitalWrite(46, LOW);     // BR
  analogWrite(9,backRightspd);
  digitalWrite(4, HIGH);      //FR
  digitalWrite(5, LOW);     // FR
  analogWrite(3,frontRightspd); //
          
  digitalWrite(11, LOW);     // FL 
  digitalWrite(12, HIGH);    // FL 
  analogWrite(10,frontLeftspd);
  digitalWrite(6, LOW);     // BL
  digitalWrite(2, HIGH);    // BL
  analogWrite(13,backLeftspd);  //다리 병신인놈임 다른놈보다 좀더 높혀야댐 다른놈들보다 30은 높혀야함
}

void moveForwardCli() 
{
  Serial.println("motorForwardCli.");
  digitalWrite(44, HIGH);      // BR
  digitalWrite(46, LOW);     // BR
  analogWrite(9,backRightspd);
  digitalWrite(4, HIGH);      //FR
  digitalWrite(5, LOW);     // FR
  analogWrite(3,frontRightspd); //
          
  digitalWrite(11, LOW);     // FL 
  digitalWrite(12, HIGH);    // FL 
  analogWrite(10,spd);
  digitalWrite(6, LOW);     // BL
  digitalWrite(2, HIGH);    // BL
  analogWrite(13,backLeftspd);  //다리 병신인놈임 다른놈보다 좀더 높혀야댐 다른놈들보다 30은 높혀야함
}

void moveLeft() //평행 왼쪽 이동
{
  Serial.println("motorLeft");
  digitalWrite(44, LOW);      // BR
  digitalWrite(46, HIGH);     // BR
  analogWrite(9, leftmovebackrightspd);
  digitalWrite(4, HIGH);      // FR
  digitalWrite(5, LOW);     // FR
  analogWrite(3,leftmovefrontrightspd);
          
  digitalWrite(11, HIGH);   // FL 
  digitalWrite(12, LOW);    // FL 
  analogWrite(10,leftmovefrontleftspd);
  digitalWrite(6, LOW);    // BL
  digitalWrite(2, HIGH);   // BL 
  analogWrite(13,leftmovebackleftspd);
}
void moveRight() //평행 우측 이동
{
  Serial.println("motorRight.");
  digitalWrite(44, HIGH);      // BR
  digitalWrite(46, LOW);     // BR
  analogWrite(9,rightmovebackrightspd);
  digitalWrite(4, LOW);      // FR
  digitalWrite(5, HIGH);     // FR
  analogWrite(3,rightmovefrontrightspd);
          
  digitalWrite(11, LOW);   // FL
  digitalWrite(12, HIGH);    // FL
  analogWrite(10,rightmovefrontleftspd); 
  digitalWrite(6, HIGH);    // BL
  digitalWrite(2, LOW);   // BL
  analogWrite(13,rightmovebackleftspd);
}

void stopp()
{
  Serial.println("motorStopp");
  digitalWrite(44, LOW);      // BR
  digitalWrite(46, LOW);     // BR
  analogWrite(9, 0);
  digitalWrite(4, LOW);      // FR
  digitalWrite(5, LOW);     // FR
  analogWrite(3, 0);
          
  digitalWrite(11, LOW);   // FL
  digitalWrite(12, LOW);    // FL 
  analogWrite(10,0);
  digitalWrite(6, LOW);    // BL 
  digitalWrite(2, LOW);   // BL
  analogWrite(13,0);
}

void movestopp() // 움직임 도중에 멈추는것 관성 생각 ㅇㅇ (뒤로 좀 주고 멈추기)
{
  Serial.println("moveStopp");
// 뒤로 잠시 이동
  digitalWrite(44, LOW);     // BR
  digitalWrite(46, HIGH);    // BR
  analogWrite(9,60);            
  digitalWrite(4, LOW);     // FR
  digitalWrite(5, HIGH);    // FR
  analogWrite(3,60); 
          
  digitalWrite(11, HIGH);    // FL
  digitalWrite(12, LOW);     // FL  
  analogWrite(10,60);           
  digitalWrite(6, HIGH);    // BL
  digitalWrite(2, LOW);     // BL
  analogWrite(13,60);

  delay(100);
  
  digitalWrite(44, LOW);      // BR
  digitalWrite(46, LOW);     // BR
  analogWrite(9, 0);
  digitalWrite(4, LOW);      // FR
  digitalWrite(5, LOW);     // FR
  analogWrite(3, 0);
          
  digitalWrite(11, LOW);   // FL
  digitalWrite(12, LOW);    // FL 
  analogWrite(10,0);
  digitalWrite(6, LOW);    // BL 
  digitalWrite(2, LOW);   // BL
  analogWrite(13,0);
}

void moveBackward() 
{
  Serial.println("motorBackward.");
  digitalWrite(44, LOW);     // BR
  digitalWrite(46, HIGH);    // BR
  analogWrite(9,moveBackBackRightspd);            
  digitalWrite(4, LOW);     // FR
  digitalWrite(5, HIGH);    // FR
  analogWrite(3,moveBackFrontRightspd); 
          
  digitalWrite(11, HIGH);    // FL
  digitalWrite(12, LOW);     // FL  
  analogWrite(10,moveBackFrontLeftspd);           
  digitalWrite(6, HIGH);    // BL
  digitalWrite(2, LOW);     // BL
  analogWrite(13,moveBackBackLeftspd);

}

void moveBackwardCli() 
{
  Serial.println("motorBackwardCli.");
  digitalWrite(44, LOW);     // BR
  digitalWrite(46, HIGH);    // BR
  analogWrite(9,clispd);            
  digitalWrite(4, LOW);     // FR
  digitalWrite(5, HIGH);    // FR
  analogWrite(3,clispd); 
          
  digitalWrite(11, HIGH);    // FL
  digitalWrite(12, LOW);     // FL  
  analogWrite(10,clispd);           
  digitalWrite(6, HIGH);    // BL
  digitalWrite(2, LOW);     // BL
  analogWrite(13,clispd);

}

void turnLeft() //축 좌측이동
{
  Serial.println("motorLeft");
  digitalWrite(44, HIGH);      // BR
  digitalWrite(46, LOW);     // BR
  analogWrite(9, Leftmovebackrightspd);
  digitalWrite(4, HIGH);      // FR
  digitalWrite(5, LOW);     // FR
  analogWrite(3, Leftmovefrontrightspd);
          
  digitalWrite(11, HIGH);   // FL 
  digitalWrite(12, LOW);    // FL 
  analogWrite(10,Leftmovefrontleftspd);
  digitalWrite(6, LOW);    // BL 
  digitalWrite(2, LOW);   // BL
  analogWrite(13,Leftmovebackleftspd);
}

void turnRight() //축 우측이동
{
  Serial.println("motorRight");
  digitalWrite(44, LOW);      // BR
  digitalWrite(46, LOW);     // BR
  analogWrite(9,Rightmovebackrightspd);
  digitalWrite(4,LOW );      // FR
  digitalWrite(5,HIGH);     // FR
  analogWrite(3,Rightmovefrontrightspd);
          
  digitalWrite(11, LOW);   // FL
  digitalWrite(12, HIGH);    // FL 
  analogWrite(10,Rightmovefrontleftspd);
  digitalWrite(6, LOW);    // BL 
  digitalWrite(2, HIGH);   // BL
  analogWrite(13,Rightmovebackleftspd);
}
