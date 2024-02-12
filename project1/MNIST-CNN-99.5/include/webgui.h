/***************************************************************************/
/* this header file is for                                                 */
/* WEBGUI A web browser based graphical user interface                     */
/* Version: 1.0 - June 25 2017                                             */
/***************************************************************************/
/* Copyright (c) 2017, Christopher Deotte                                  */
/* Funding: NSF DMS-1345013                                                */
/* Documentation: http://ccom.ucsd.edu/~cdeotte/webgui                     */
/***************************************************************************/

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
void websetmode(int x);