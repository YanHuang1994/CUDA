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
/* this utility program helps in modifying                                 */
/* WEBGUI A web browser based graphical user interface                     */
/* Version: 1.0 - June 25 2017                                             */
/***************************************************************************/
/* Author: Christopher Deotte                                              */
/* Advisor: Randolph E. Bank                                               */
/* Funding: NSF DMS-1345013                                                */
/* Documentation: http://ccom.ucsd.edu/~cdeotte/webgui                     */
/***************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char* argv[]){
    printf("\nThis program converts \"indexC.html\" to an array\n");
    printf("of C strings and writes the output to \"output.txt\".\n");
    printf("To convert a different input file and/or output file\n");
    printf("usage: convert2str source_file target_file\n\n");
    int maxf = 1000000;
    unsigned char data[maxf];
    char filetemp[maxf];
    char fileinput[80]="indexC.html";
    char fileoutput[80]="output.txt";
    char dataname[80];
    if (argc>=2) strcpy(fileinput,argv[1]);
    if (argc>=3) strcpy(fileoutput,argv[2]);
    int size=0, i, j=0;
    if (access(fileinput,F_OK)!=0){
        printf("ERROR: cannot find file \"%s\".\n",fileinput);
        return -1;
    }
    
    /* Read file */
    FILE *fp;
    fp = fopen(fileinput,"r");
    size = fread(data,1,maxf,fp);
    if (size==maxf) {
        printf("ERROR: increase buffer size. File \"%s\" is larger than %d bytes.\n",fileinput,maxf);
        return -1;
    }
    printf("Read %d bytes from \"%s\"\n",size,fileinput);
    fclose(fp);
    
    /* Write file to buffer */
    char buffer[1024];
    int index=0, lines=0, max=0, len=0;
    for (i=0;i<size;i++){
        if (data[i]!='\n'){
            if (data[i]!='"' && data[i]!='\r') buffer[index++] = data[i];
            else if (data[i]!='\r') {
                buffer[index++] = '\\';
                buffer[index++] = '"';
            }
            if (data[i]=='\\') buffer[index++] = '\\';
        }
        else{
            buffer[index]='\0';
            len += sprintf(filetemp+len,"%s\",\n    \"",buffer);
            if (index>max) max = index;
            index = 0;
            lines++;
        }
    }
    buffer[index]='\0';
    len += sprintf(filetemp+len,"%s\"\n};",buffer);
    if (index>max) max = index;
    lines++;

    /* Write output */
    for (i=0;i<strlen(fileinput);i++)
        if (fileinput[i]!='.') j += sprintf(dataname+j,"%c",fileinput[i]);
    fp = fopen(fileoutput,"w+");
    fprintf(fp,"char %s[%d][%d] = {\n    \"",dataname,lines,max);
    fprintf(fp,"%s",filetemp);
    fclose(fp);
    
    printf("Wrote %d lines with max = %d to \"%s\"\n\n",lines,max,fileoutput);
    return 0;
}