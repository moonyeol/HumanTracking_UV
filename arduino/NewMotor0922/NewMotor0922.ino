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

      case 'R':
      Serial.println("turnRight");
      turnRight();
      break;

      case 'L':
      Serial.println("turnLeft");
      turnLeft();
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
  analogWrite(9,60);
  digitalWrite(4, HIGH);      //FR
  digitalWrite(5, LOW);     // FR
  analogWrite(3,60); //50 너무 느려
          
  digitalWrite(11, LOW);     // FL 
  digitalWrite(12, HIGH);    // FL 
  analogWrite(10,65);
  digitalWrite(6, LOW);     // BL
  digitalWrite(2, HIGH);    // BL
  analogWrite(13,78);  //다리 병신인놈임 다른놈보다 좀더 높혀야댐 다른놈들보다 30은 높혀야함
}
void moveLeft() 
{
  Serial.println("motorLeft");
  digitalWrite(44, LOW);      // BR
  digitalWrite(46, HIGH);     // BR
  analogWrite(9, 75);
  digitalWrite(4, HIGH);      // FR
  digitalWrite(5, LOW);     // FR
  analogWrite(3, 75);
          
  digitalWrite(11, HIGH);   // FL 
  digitalWrite(12, LOW);    // FL 
  analogWrite(10,75);
  digitalWrite(6, LOW);    // BL
  digitalWrite(2, HIGH);   // BL 
  analogWrite(13,93);
}
void moveRight() 
{
  Serial.println("motorRight.");
  digitalWrite(44, HIGH);      // BR
  digitalWrite(46, LOW);     // BR
  analogWrite(9,75);
  digitalWrite(4, LOW);      // FR
  digitalWrite(5, HIGH);     // FR
  analogWrite(3,75);
          
  digitalWrite(11, LOW);   // FL
  digitalWrite(12, HIGH);    // FL
  analogWrite(10,75); 
  digitalWrite(6, HIGH);    // BL
  digitalWrite(2, LOW);   // BL
  analogWrite(13,93);
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

void moveBackward() 
{
  Serial.println("motorBackward.");
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
  analogWrite(13,78);

}

void turnLeft() 
{
  Serial.println("motorLeft");
  digitalWrite(44, HIGH);      // BR
  digitalWrite(46, LOW);     // BR
  analogWrite(9, 75);
  digitalWrite(4, HIGH);      // FR
  digitalWrite(5, LOW);     // FR
  analogWrite(3, 75);
          
  digitalWrite(11, HIGH);   // FL 
  digitalWrite(12, LOW);    // FL 
  analogWrite(10,75);
  digitalWrite(6, LOW);    // BL 
  digitalWrite(2, LOW);   // BL
  analogWrite(13,93);
}

void turnRight() 
{
  Serial.println("motorRight");
  digitalWrite(44, LOW);      // BR
  digitalWrite(46, LOW);     // BR
  analogWrite(9, 75);
  digitalWrite(4,LOW );      // FR
  digitalWrite(5,HIGH);     // FR
  analogWrite(3, 75);
          
  digitalWrite(11, LOW);   // FL
  digitalWrite(12, HIGH);    // FL 
  analogWrite(10,75);
  digitalWrite(6, LOW);    // BL 
  digitalWrite(2, HIGH);   // BL
  analogWrite(13,93);
}
