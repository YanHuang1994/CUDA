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
#include<stdlib.h>
#include<string.h>
#include<unistd.h>

int webstart(int x);
void webreadline(char* str);
void webwriteline(char* str);
void websettitle(char* str);
void websetcolors(int nc, double* r,double* g,double* b,int pane);
void webframe(int frame);
void webfillflt(float* x, float* y, float* z, int n, int color);
void webgldisplay(int pane);
void websetmode(int x);

int ct=0;
float x[100], y[100];
char buffer[80];
double r[1]={0}, g[1]={0}, b[1]={1};

void drawpoints();
char* extractVal(char* str, char key);

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
int main(int argc, char *argv[]){
    int p=0;
    char str[80];
    websetmode(2);
    websettitle("Mouse example");
    webstart(15000);
    webwriteline("Example of mouse capture.");
    webwriteline("Click mouse in top right display pane.");
    while(1){
        webreadline(str);
        if (str[0]=='q') exit(0);
        /* expecting str = "mse button=0,x=0.0,y=0.0,pane=0" */
        if (str[0]=='m'){
            if (str[11]=='0' && atoi(extractVal(str+4,'e'))==0){
            /* if button=0 and pane=0 */
                x[ct] = 0.5 + atof(extractVal(str,'x'))/2.0;
                y[ct] = 0.5 + atof(extractVal(str,'y'))/2.0;
                ct++;
                drawpoints();
            }
        }
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
void drawpoints(){
    int i;
    float xx[4], yy[4];
    float zz[4]={0.5,0.5,0.5,0.5};
    websetcolors(1,r,g,b,0);
    webframe(5);
    for (i=0;i<ct;i++){
        xx[0] = x[i]-0.02; xx[1] = x[i];
        xx[2] = x[i]+0.02; xx[3] = x[i];
        yy[0] = y[i]; yy[1] = y[i]+0.02;
        yy[2] = y[i]; yy[3] = y[i]-0.02;
        webfillflt(xx,yy,zz,4,1);
    }
    webgldisplay(0);
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
char* extractVal(char* str, char key){
    /* returns the value associated with key in str */
    buffer[0]=0;
    int index1 = 0, index2;
    while (index1<strlen(str)){
        if (str[index1]=='='){
            if (str[index1-1]==key){
                index2 = index1;
                while (index2<strlen(str) && str[index2]!=',') index2++;
                strncpy(buffer,str+index1+1,index2-index1-1);
                buffer[index2-index1-1]=0;
                break;
            }
        }
        index1++;
    }
    return buffer;
}