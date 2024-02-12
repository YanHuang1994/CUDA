/**********************************************************************************/
/* Copyright (c) 2017, Christopher Deotte                                         */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files (the "Software"), to deal  */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

/***************************************************************************/
/* this program is an example driver for                                   */
/* WEBGUI A web browser based graphical user interface                     */
/* Version: 1.0 - June 25 2017                                             */
/***************************************************************************/
/* Author: Christopher Deotte                                              */
/* Advisor: Randolph E. Bank                                               */
/* Funding: NSF DMS-1345013                                                */
/* Documentation: http://ccom.ucsd.edu/~cdeotte/webgui                     */
/***************************************************************************/

#include<stdio.h>
#include<unistd.h>
#include<pthread.h>

int webstart(int x);
void webreadline(char* str);
void webwriteline(char* str);
void websettitle(char* str);
void websetcolors(int nc, double* r,double* g,double* b,int pane);
void webimagedisplay(int nx, int ny, int* image, int ipane);
void websetmode(int x);

pthread_t pth;
int image[480][720]={{0}};
int x=336, y=216, d=48;
double r[2]={1,1}, g[2]={1,0}, b[2]={1,0};
int v[2]={6,0}, stop=0;

void *processcommand(void *arg);
void clearimage();
void drawrectangle();

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
int main(int argc, char *argv[]){
    websetmode(6);
    websettitle("Keyboard example");
    webstart(15000);
    webwriteline("Example of animation and keyboard capture.");
    webwriteline("Use arrow keys to change direction.");
    websetcolors(2,r,g,b,3);
    pthread_create(&pth,NULL,processcommand,NULL);
    while(stop==0){
        clearimage();
        drawrectangle();
        webimagedisplay(720,480,(int*)image,3);
        x+=v[0]+720; x%=720;
        y+=v[1]+480; y%=480;
    }
    return 0;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void *processcommand(void *arg){
    char str[80];
    while(1){
        webreadline(str);
        if (str[0]=='q') stop=1;
        /* expecting str = "key code=38" */
        else if (str[0]=='k'){
            if (str[10]=='8') {v[0]=0; v[1]=6;} // up is 38
            if (str[10]=='0') {v[0]=0; v[1]=-6;} // down is 40
            if (str[10]=='7') {v[0]=-6; v[1]=0;} // left is 37
            if (str[10]=='9') {v[0]=6; v[1]=0;} // right is 39
            if (str[10]=='2') {v[0]=0; v[1]=0;} // space is 32
        }
    }
}
/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void clearimage(){
    int i,j;
    for (i=0;i<480;i++)
    for (j=0;j<720;j++)
        image[i][j]=0;
}
/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void drawrectangle(){
    int i,j;
    for (i=y;i<y+d;i++)
    for (j=x;j<x+d;j++)
        image[i%480][j%720]=1;
}