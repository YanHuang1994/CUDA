#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/stat.h>
#include<unistd.h>
#include<dirent.h>
#include<math.h>
#include<time.h>
#include<pthread.h>

/* WEBGUI External Routines needed from webgui.c */
/* (These could be placed into a webgui.h) */

/**
 * Initializes and starts the web GUI server.
 * @param x Configuration parameter for server initialization, generally 15000.
 * @return Status of the operation (0 for success, non-zero for error).
 */
int webstart(int x);

/**
 * Retrieves the oldest unread command string from the command queue.
 * If no command is available, it waits until a command is submitted from the web browser.
 * @param str Char buffer (80 chars length) to store the command. The buffer is filled with spaces for unused parts, following Fortran conventions.
 */
void webreadline(char* str);

/**
 * Displays a line of text in the web browser's graphical interface text output pane.
 * @param str Pointer to the string to be displayed. This function is responsible for
 * rendering the given text string directly in the web browser, allowing for dynamic
 * interaction with the user through the web interface.
 */
void webwriteline(char* str);

/**
 * Initializes the web GUI with specified settings.
 * @param str Configuration string.
 * @param x Additional configuration parameter.
 */
void webinit(char* str,int x);
void webupdate(int* ip, double* rp, char* sp);
void websettitle(char* str);
void websetmode(int x);
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
unsigned long fsize(char* file);

/* WEBGUI Internal Routines and Variables */
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
/* general variables for any program using webgui.c */
int ct=0;
char** map_keys;
int* map_indices;
char* map_array;
double *rp_default, *rp;
int *ip_default, *ip;
char *sp_default, *sp;
char buffer[80];

/* Program Specific Routines */
/* Organized by category. (These functions */
/* could be placed in 10 different C files.) */

// LOAD DATA
int loadTrain(int ct, double testProp, int sh, float imgScale, float imgBias);
int loadTest(int ct, int sh, int rc, float imgScale, float imgBias);
// DISPLAY DATA
void displayDigit(int x, int ct, int p, int lay, int chan, int train, int cfy, int big);
void displayDigits(int *dgs, int ct, int pane, int train, int cfy, int wd, int big);
void doAugment(int img, int big, int t);
void viewAugment(int img, int ct, float rat, int r, float sc, int dx, int dy, int p, int big, int t, int cfy);
void printDigit(int a,int b,int *canvas,int m,int n,float *digit,int row,int col,int d3,int d1,int d2);
void printInt(int a,int b,int *canvas,int m,int n,int num,int col,int d1);
void printStr(int a,int b,int *canvas,int m,int n,char *str,int col,int d1);
// ADVANCED DISPLAY
void displayFilter(int ct, int p, int lay, int chan);
void displayFilter2(int ct, int p, int lay, int chan);
void maxActivations(int ct, int p, int lay, int chan, int t, int x);
void maxActivations2(int ct, int p, int lay, int chan, int t, int x);
void maxActivations3(int ct, int p, int lay, int chan, int train, int x);
void drawBorders(int lay, int chan, int *idxA, int *idxB, int ct);
void fakeHeap3(float *dist,int *idx, int *idxA, int *idxB, int k);
void sortHeap3(float *dist, int *idx, int *idxA, int *idxB, int k);
void setColors();
void setColors2();
void setColors3();
// DISPLAY PROGRESS
void displayConfusion(int (*confusion)[10]);
void displayCDigits(int x,int y);
void displayEntropy(float *ents, int entSize, float *ents2, int display);
void displayAccuracy(float *accs, int accSize,float *accs2, int display);
void line(int* img, int m, int n,float x1, float y1, float x2, float y2,int d,int c);
// DISPLAY DOTS
void clearImage(int p);
void updateImage();
void displayClassify(int dd);
void displayClassify3D();
void setColors4();
void placeDots();
void removeDot(float x, float y);
// INIT-NET
void initNet(int t);
void initArch(char *str, int x);
// NEURAL-NET
int isDigits(int init);
void randomizeTrainSet();
void dataAugment(int img, int r, float sc, float dx, float dy, int p, int hiRes, int loRes, int t);
void *runBackProp(void *arg);
int backProp(int x,float *ent, int ep);
int forwardProp(int x, int dp, int train, int lay);
float ReLU(float x);
float TanH(float x);
// KNN
void *runKNN(void *arg);
int singleKNN(int x, int k, int d, int p, int train, int big, int disp);
void fakeHeap(float *dist,int *idx,int k);
void sortHeap(float *dist,int *idx,int k);
float distance(float *digitA, float *digitB, int n, int x);
// PREDICT
void *predictKNN(void *arg);
void writePredictFile(int NN, int k, int d, int y, int big);
void writeFile();
// SPECIAL AND/OR DEBUG
void dreamProp(int y, int it, float bs, float ft, int ds, int lay, int chan, int p);
void dream(int x, int y, int it, float bs, float ft, int ds, int lay, int chan, int p);
int heatmap(int x, int t, int p, int wd);
void boundingBoxes();
void initData(int z);
void displayWeights();
void writeFile2();
float square(float x);