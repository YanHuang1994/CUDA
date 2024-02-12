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

/* External Routines needed from webgui.c */
/* (these can be placed into a webgui.h) */
int webstart(int x);
void webreadline(char* str);
void webwriteline(char* str);
void webinit(char* str,int x);
void webupdate(int* ip, double* rp, char* sp);
void websettitle(char* str);
void webstop();
void websetcolors(int nc, double* r,double* g,double* b,int pane);
void webimagedisplay(int nx, int ny, int* image, int ipane);
void webframe(int frame);
void weblineflt(float* x, float* y, float* z, int n, int color);
void webfillflt(float* x, float* y, float* z, int n, int color);
void weblinedbl(double* x, double* y, double* z, int n, int color);
void webfilldbl(double* x, double* y, double* z, int n, int icolor);
void webgldisplay(int pane);
int webquery();
void webbutton(int highlight,char* cmd);
void webpause();

/* Internal Routines */
/* general routines for any program using webgui.c */
void initParameterMap(char* str, int n);
char* extractVal(char* str, char key);
char processCommand(char* str);
void updateParameter(char* str, int index1, int index2);
char arrayGet(char* key);
int ipGet(char* key);
void ipSet(char* key, int value);
double rpGet(char* key);
void rpSet(char* key, double value);
char* spGet(char* key);
void spSet(char* key, char* value);
/* general variables */
int ct=0;
char** map_keys;
int* map_indices;
char* map_array;
double *rp_default, *rp;
int *ip_default, *ip;
char *sp_default, *sp;
char buffer[80];

/* program specific routines */
void drawTriangle();
void drawTriangleOutline();
void drawLine(int display);
void drawAllObjects(int pane);
void clearDisplay();
/* program specific variables */
char init[60][80]={
    "c c=DrawTriangle,k=t",
    "c c=ToggleFill, k=f",
    "c c=DrawLine,k=l",
    "c c=ClearDisplayPane, k=c",
    "c c=ResetParameters, k=r",
    "c c=Quit, k=q",
    "n n=x1, t=r, i=1, d=0.25",
    "n n=y1, t=r, i=2, d=0.25",
    "n n=z1, t=r, i=3, d=0.5",
    "n n=x2, t=r, i=4, d=0.75",
    "n n=y2, t=r, i=5, d=0.25",
    "n n=z2, t=r, i=6, d=0.5",
    "n n=x3, t=r, i=7, d=0.5",
    "n n=y3, t=r, i=8, d=0.75",
    "n n=z3, t=r, i=9, d=0.5",
    "n n=x4, t=r, i=10, d=0.25",
    "n n=y4, t=r, i=11, d=0.5",
    "n n=z4, t=r, i=12, d=0.5",
    "n n=x5, t=r, i=13, d=0.75",
    "n n=y5, t=r, i=14, d=0.5",
    "n n=z5, t=r, i=15, d=0.5",
    "n n=Red, t=i, i=1, d=0",
    "n n=Green, t=i, i=2, d=0",
    "n n=Blue, t=i, i=3, d=0",
    "n n=frame, t=i, i=4, d=5",
    "n n=pane, t=i, i=5, d=0",
    "r c=DrawTriangle, n=x1",
    "r c=DrawTriangle, n=y1",
    "r c=DrawTriangle, n=z1",
    "r c=DrawTriangle, n=x2",
    "r c=DrawTriangle, n=y2",
    "r c=DrawTriangle, n=z2",
    "r c=DrawTriangle, n=x3",
    "r c=DrawTriangle, n=y3",
    "r c=DrawTriangle, n=z3",
    "r c=DrawTriangle, n=Red",
    "r c=DrawTriangle, n=Green",
    "r c=DrawTriangle, n=Blue",
    "r c=DrawTriangle, n=frame",
    "r c=DrawTriangle, n=pane",
    "r c=DrawLine, n=x4",
    "r c=DrawLine, n=y4",
    "r c=DrawLine, n=z4",
    "r c=DrawLine, n=x5",
    "r c=DrawLine, n=y5",
    "r c=DrawLine, n=z5",
    "r c=DrawLine, n=Red",
    "r c=DrawLine, n=Green",
    "r c=DrawLine, n=Blue",
    "r c=DrawLine, n=frame",
    "r c=DrawLine, n=pane",
    "r c=ClearDisplayPane, n=pane",
    "s n=pane, v=0, l=\"0 top right\"",
    "s n=pane, v=1, l=\"1 bottom left\"",
    "s n=pane, v=2, l=\"2 bottom right\"",
    "s n=frame, v=1, l=\"1 all\"",
    "s n=frame, v=2, l=\"2 top right\"",
    "s n=frame, v=3, l=\"3 bottom right\"",
    "s n=frame, v=4, l=\"4 left\"",
    "s n=frame, v=5, l=\"5 rotate\""
};
float triangles[3][1000], lines[3][700];
int indexT[3]={0,0,0}, indexL[3]={0,0,0};
double red[3][200], green[3][200], blue[3][200];
int fill=1;

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
int main(int argc, char *argv[]){
    int i, offset=0;
    char cmd, str[80];
    initParameterMap((char*)init,60);
    webinit((char*)init,60);
    websettitle("Sample WEBGUI driver");
    while (webstart(15000+offset)<0) offset++;
    while (1){
        webreadline(str);
        cmd = processCommand(str);
        if (cmd=='t'){
            if (fill==1){
                drawTriangle();
                webwriteline("-Triangle drawn with fill.");
            }
            else{
                drawTriangleOutline();
                webwriteline("-Triangle drawn without fill.");
            }
        }
        else if (cmd=='f'){
            fill *= -1;
            if (fill==1){
                webwriteline("-Triangles will be filled.");
                webbutton(0,"ToggleFill");
            }
            else {
                webwriteline("-Triangles will not be filled.");
                webbutton(1,"ToggleFill");
            }
        }
        else if (cmd=='l'){
            drawLine(1);
            webwriteline("-Line drawn.");
        }
        else if (cmd=='c'){
            clearDisplay();
            webwriteline("-Display pane cleared.");
        }
        else if (cmd=='r'){
            webupdate(ip_default,rp_default,sp_default);
            for (i=0;i<ct;i++){
                ip[i] = ip_default[i];
                rp[i] = rp_default[i];
                strcpy(sp+80*i, sp_default+80*i);
            }
            webwriteline("-Parameters reset.");
        }
        else if (cmd=='q'){
            webwriteline("-Quitting.");
            webstop();
            return 0;
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
void initParameterMap(char* str, int n){
    /* reads array of strings and initializes ip, rp, sp */
    /* and creates a map for accessing ip, rp, and sp */
    int i, index=0;
    for (i=0; i<n; i++) if (str[80*i]=='n') ct++;
    map_keys = (char**)malloc(ct * sizeof(char*));
    map_indices = (int*)malloc(ct * sizeof(int));
    map_array = (char*)malloc(ct * sizeof(char));
    rp_default = (double*)malloc(ct * sizeof(double));
    ip_default = (int*)malloc(ct * sizeof(int));
    sp_default = (char*)malloc(ct * sizeof(char*) * 80);
    rp = (double*)malloc(ct * sizeof(double));
    ip = (int*)malloc(ct * sizeof(int));
    sp = (char*)malloc(ct * sizeof(char*) * 80);
    for (i=0; i<ct; i++) map_keys[i] = (char*)malloc(20 * sizeof(char));
    for (i=0; i<n; i++)
    if (str[80*i]=='n'){
        strcpy(map_keys[index],extractVal(str+80*i,'n'));
        map_indices[index] = atoi(extractVal(str+80*i,'i'))-1;
        map_array[index] = *extractVal(str+80*i,'t');
        if (map_array[index]=='r'){
            rp_default[map_indices[index]] = atof(extractVal(str+80*i,'d'));
            rp[map_indices[index]] = rp_default[map_indices[index]];
        }
        else if (map_array[index]=='i'){
            ip_default[map_indices[index]] = atoi(extractVal(str+80*i,'d'));
            ip[map_indices[index]] = ip_default[map_indices[index]];
        }
        else if (map_array[index]=='s' || map_array[index]=='f' || map_array[index]=='l'){
            strcpy(sp_default+80*map_indices[index],extractVal(str+80*i,'d'));
            strcpy(sp+80*map_indices[index],sp_default+80*map_indices[index]);
        }
        index++;
    }
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
    str[79]=0;
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

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
char processCommand(char* str){
    /* returns command char and updates parameters */
    int index1 = 1, index2 = 2;
    while (str[index2]!=' '){
        if (str[index2]==','){
            updateParameter(str,index1,index2);
            index1 = index2;
        }
        index2++;
    }
    if (index2>2) updateParameter(str,index1,index2);
    return str[0];
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void updateParameter(char* str, int index1, int index2){
    /* parses str between index1 and index2 and updates parameter */
    int index3 = index1+1;
    while (str[index3]!='=') index3++;
    str[index2]=0; str[index3]=0;
    char ch = arrayGet(str+index1+1);
    if (ch=='r') rpSet(str+index1+1,atof(str+index3+1));
    else if (ch=='i') ipSet(str+index1+1,atoi(str+index3+1));
    else if (ch=='s' || ch=='f' || ch=='l') spSet(str+index1+1,str+index3+1);
    str[index2]=','; str[index3]='=';
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
char arrayGet(char* key){
    /* returns which array (ip, rp, sp) key belongs to */
    int i;
    char value = ' ';
    for (i=0; i<ct; i++) if (strcmp(map_keys[i],key)==0)
        value = map_array[i];
    return value;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
int ipGet(char* key){
    int i, value = 0;
    for (i=0; i<ct; i++) if (strcmp(map_keys[i],key)==0)
        value = ip[map_indices[i]];
    return value;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void ipSet(char* key, int value){
    int i;
    for (i=0; i<ct; i++) if (strcmp(map_keys[i],key)==0)
        ip[map_indices[i]] = value;
    return;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
double rpGet(char* key){
    int i;
    double value = 0;
    for (i=0; i<ct; i++) if (strcmp(map_keys[i],key)==0)
        value = rp[map_indices[i]];
    return value;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void rpSet(char* key, double value){
    int i;
    for (i=0; i<ct; i++) if (strcmp(map_keys[i],key)==0)
        rp[map_indices[i]] = value;
    return;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
char* spGet(char* key){
    int i;
    buffer[0] = 0;
    for (i=0; i<ct; i++) if (strcmp(map_keys[i],key)==0)
        strcpy(buffer,sp + 80 * map_indices[i]);
    return buffer;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void spSet(char* key, char* value){
    int i;
    for (i=0; i<ct; i++) if (strcmp(map_keys[i],key)==0)
        strcpy(sp + 80 * map_indices[i],value);
    return;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void drawTriangle(){
    int i, j;
    char str[3], var[3]={'x','y','z'};
    int pane = ipGet("pane");
    /* save triangle data locally */
    triangles[pane][indexT[pane]*10 + 9] = ipGet("frame");
    for (int i=1;i<4;i++)
    for (int j=0;j<3;j++){
        sprintf(str,"%c%d",var[j],i);
        triangles[pane][indexT[pane]*10+3*(i-1)+j] = (float)rpGet(str);
    }
    red[pane][indexT[pane]] = ipGet("Red")/255.0;
    green[pane][indexT[pane]] = ipGet("Green")/255.0;
    blue[pane][indexT[pane]] = ipGet("Blue")/255.0;
    indexT[pane]++;
    /* draw all triangles and lines to pane */
    drawAllObjects(pane);
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void drawTriangleOutline(){
    int i, j;
    double x4=rpGet("x4"), y4=rpGet("y4"), z4=rpGet("z4");
    double x5=rpGet("x5"), y5=rpGet("y5"), z5=rpGet("z5");
    char str[3], str2[3], var[3]={'x','y','z'};
    for (i=4;i<6;i++)
    for (j=0;j<3;j++){
        sprintf(str,"%c%d",var[j],i);
        sprintf(str2,"%c%d",var[j],i-3);
        rpSet(str, rpGet(str2));
    }
    drawLine(0);
    for (i=4;i<6;i++)
    for (j=0;j<3;j++){
        sprintf(str,"%c%d",var[j],i);
        sprintf(str2,"%c%d",var[j],i-2);
        rpSet(str, rpGet(str2));
    }
    drawLine(0);
    for (j=0;j<3;j++){
        sprintf(str,"%c4",var[j]);
        sprintf(str2,"%c1",var[j]);
        rpSet(str, rpGet(str2));
    }
    for (j=0;j<3;j++){
        sprintf(str,"%c5",var[j]);
        sprintf(str2,"%c3",var[j]);
        rpSet(str, rpGet(str2));
    }
    drawLine(1);
    rpSet("x4",x4); rpSet("y4",y4); rpSet("z4",z4);
    rpSet("x5",x5); rpSet("y5",y5); rpSet("z5",z5);
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void drawLine(int display){
    int i, j;
    char str[3], var[3]={'x','y','z'};
    int pane = ipGet("pane");
    /* save line data locally */
    lines[pane][indexL[pane]*7 + 6] = ipGet("frame");
    for (int i=4;i<6;i++)
    for (int j=0;j<3;j++) {
        sprintf(str,"%c%d",var[j],i);
        lines[pane][indexL[pane]*7+3*(i-4)+j] = (float)rpGet(str);
    }
    red[pane][100+indexL[pane]] = ipGet("Red")/255.0;
    green[pane][100+indexL[pane]] = ipGet("Green")/255.0;
    blue[pane][100+indexL[pane]] = ipGet("Blue")/255.0;
    indexL[pane]++;
    /* draw all triangles and lines to pane */
    if (display==1) drawAllObjects(pane);
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void drawAllObjects(int pane){
    int i, j;
    float x[3], y[3], z[3];
    websetcolors(200,red[pane],green[pane],blue[pane],pane);
    for (i=0;i<indexT[pane];i++){
        webframe((int)triangles[pane][i*10+9]);
        for (j=0;j<3;j++){
            x[j] = triangles[pane][10*i+3*j];
            y[j] = triangles[pane][10*i+3*j+1];
            z[j] = triangles[pane][10*i+3*j+2];
        }
        webfillflt(x,y,z,3,i+1);
    }
    for (i=0;i<indexL[pane];i++){
        webframe((int)lines[pane][i*7+6]);
        for (j=0;j<2;j++){
            x[j] = lines[pane][7*i+3*j];
            y[j] = lines[pane][7*i+3*j+1];
            z[j] = lines[pane][7*i+3*j+2];
        }
        weblineflt(x,y,z,2,i+101);
    }
    webgldisplay(pane);
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void clearDisplay(){
    int pane = ipGet("pane");
    indexL[pane]=0;
    indexT[pane]=0;
    websetcolors(200,red[pane],green[pane],blue[pane],pane);
    webgldisplay(pane);
}
