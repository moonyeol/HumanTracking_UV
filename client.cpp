#include "socket.h"
int main(int argc, char* argv[])
{
    /*
    if(argc != 2)
    {    // argv[0]에 ./client가 들어간다.
    printf("사용법 : %s IPv4-address\n", argv[0]);
    return -1;
    }
    */
 
    int client_socket;
    struct sockaddr_in serverAddress;
    unsigned int server_addr_size;
    char sendBuff[BUFF_SIZE];
    char readBuff[BUFF_SIZE];
 
 
    ssize_t receivedBytes;
    ssize_t sentBytes;
 
 
    memset(&serverAddress, 0, sizeof(serverAddress));
 
    serverAddress.sin_family = AF_INET;
    inet_aton("127.0.0.1", (struct in_addr*) &serverAddress.sin_addr.s_addr);
    serverAddress.sin_port = htons(20162);
 
    // 소켓 생성
    if ((client_socket = socket(PF_INET, SOCK_DGRAM, 0)) == -1)
    {
        printf("socket 생성 실패\n");
        exit(0);
    }
 
    while (1)
    {
        // 채팅 프로그램 제작
 
        server_addr_size = sizeof(serverAddress);
        //클라이언트에서 메세지 전송
        printf("클라이언트에서 보낼 말을 입력하세요 :: ");
 
        char msg[BUFF_SIZE];
        fgets(msg, BUFF_SIZE, stdin);
 
        sprintf(sendBuff, "%s", msg);
 
        sentBytes = sendto(client_socket, sendBuff, strlen(sendBuff), 0, (struct sockaddr*)&serverAddress, sizeof(serverAddress));
 
 
        receivedBytes = recvfrom(client_socket, readBuff, BUFF_SIZE, 0, (struct sockaddr*)&serverAddress, &server_addr_size);
        printf("%lu bytes read\n", receivedBytes);
        readBuff[receivedBytes] = '\0';
        fputs(readBuff, stdout);
        fflush(stdout);
 
    }
 
    // 소켓 close
    close(client_socket);
    return 0;
} 

