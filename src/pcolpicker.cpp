// =====================================================================
//  pcolpicker - PricolorPicker
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//    get primary color from image file.
//
//    This program picks up primary color of image file.
//    In this program, primary color is defined as a chromatich color
//    in image.
//
//    The process will go with several steps.
//    First, PricolorPicker will try to find a most used color that
//    chromatic value is more than 0.5.
//    If it can't find it, program will try to find one more than 0.2.
//    If it can't still, program will output most used color.
//
//    ---
//    Developed by:
//        T.Onodera <onodera.takahiro@adways.net>
//    Version:
//        1.0.0
//    Cf.:
//        http://aidiary.hatenablog.com/entry/20091003/1254574041
//
//    (C)ADWAYS Inc. all rights reserved.
// =====================================================================

#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <getopt.h>
#include <fcntl.h>
#include <unistd.h> 
#include <sys/types.h>
#include <sys/stat.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"

#define MAX3(a, b, c) ((a) > (MAX(b, c)) ? (a) : (MAX(b, c)))
#define MIN3(a, b, c) ((a) < (MIN(b, c)) ? (a) : (MIN(b, c)))

// HSV Model, CONIC / COLUMNAR
#define USE_CONIC_MODEL 1

// Filesize limitation
#define MAX_FILESIZE    ((unsigned long)1000000000)

// Color depth(MAX 8); 2^COLDEPTH, 2^4->16, 16^3->4096
#define COLDEPTH        4
#define NUM_OF_BIN      (1 << (COLDEPTH * 3))

// Default Chrome border line
#define DEF_UCHROME     0.5
#define DEF_LCHROME     0.2


using namespace std;


int calcHistogram(char *, int *, float *);
int rgb2bin(int, int, int);
void bin2rgb(int, int *);
float calc_chrome(int, int, int);
long get_filesize(const char *);
static void help(char *);


/**
 * Calculate histgram of image
 *
 * @param[in]  filename   image file
 * @param[out] histogram  histgram data
 * @return ok ... 0, fault ... -1
 */
int calcHistogram(char *filename, int histogram[NUM_OF_BIN], float chromes[NUM_OF_BIN] )
{
	// load image
	IplImage *img = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);

	if (img == NULL) {
		cerr << "Cannot open image file: " << filename << endl;
		return -1;
	}

	for (int y = 0; y < img->height; y++)
	{
		uchar *pin = (uchar *)(img->imageData + y * img->widthStep);
		for (int x = 0; x < img->width; x++)
		{
			int b = pin[ x*3 +0 ];
			int g = pin[ x*3 +1 ];
			int r = pin[ x*3 +2 ];

			int bin = rgb2bin( r, g, b );

			// note: If want color tone adjusting,
			// changing addition value of histgram (should be float)
			// wiil be effective...maybe. 
			histogram[ bin ] += 1;
			if ( !chromes[bin] ) chromes[bin] = calc_chrome( r, g, b );
		}
	}

	cvReleaseImage(&img);
	return 0;
}


/**
 * Get bin number from (r,g,b)
 *
 * @param[in]  (r,g,b)    RGB values
 * @return bin no.
 */
int rgb2bin(int r, int g, int b)
{
	int r0 = r >> (8-COLDEPTH);
	int g0 = g >> (8-COLDEPTH);
	int b0 = b >> (8-COLDEPTH);
	return ( r0 << (COLDEPTH*2) ) + ( g0 << COLDEPTH ) + b0;
}

void bin2rgb(int bin, int rgb[3])
{
	int r,g,b;
	int cmpl = (1<<COLDEPTH) -1;

	b = bin & cmpl;
	bin >>= COLDEPTH;
	g = bin & cmpl;
	bin >>= COLDEPTH;
	r = bin & cmpl;

	rgb[0] = r;
	rgb[1] = g;
	rgb[2] = b;
}

float calc_chrome(int r, int g, int b)
{
	int r0 = r >> (8-COLDEPTH);
	int g0 = g >> (8-COLDEPTH);
	int b0 = b >> (8-COLDEPTH);

	float max = MAX3(r0, g0, b0);
	float min = MIN3(r0, g0, b0);

	if ( max <= 0 || max == min ) return 0.0;

#if USE_CONIC_MODEL
	// by conic model
	int dev = (1<<COLDEPTH) -1;
	return (max - min) / dev;
#else
	// by columnar model
	return (max - min) / max;
#endif
}


// ----------------------------------------------------------------------

long get_filesize(const char *filename)
{
	int fd = open(filename, O_RDONLY);
	struct stat statBuf;
	if ( fstat(fd, &statBuf) == -1 ){
		close(fd);
		return -1;
	}
	close(fd);
	return (unsigned long) statBuf.st_size;
}

// ----------------------------------------------------------------------

int main(int argc, char **argv)
{
	int   ret;
	float crate   = DEF_UCHROME;
	float crate2  = DEF_LCHROME;
	int   peakonly= 0;
	int   hexout  = 0;
	int   sstout  = 0;
	char  *imagefile;

	while( (ret = getopt(argc, argv, "hpxsc:n:")) != -1 )
	{
		switch(ret){
			default:
			case 'h':
				help(argv[0]);
				return 255;

			case '?':
				cerr << "Unknown option." << endl;
				return -1;

			case 'c':
				crate = (float)atof(optarg);
				if ( !( crate > 0.0 && crate < 1.0 ) ){
					cerr << "Chroma border parameter must be 0.0 < n < 1.0." << endl;
					return -1;
				}
				break;

			case 'u':
				crate2 = (float)atof(optarg);
				if ( !( crate2 > 0.0 && crate2 < 1.0 ) ){
					cerr << "Chroma(2nd) border parameter must be 0.0 < n < 1.0." << endl;
					return -1;
				}
				break;

			case 'p':
				peakonly = 1;
				break;

			case 'x':	// hex output
				hexout = 1;
				break;

			case 's':	// hex output
				sstout = 1;
				break;
		}
	}

//	if ( crate2 >= crate ){
//		cerr << "Second chroma border param is bigger or equarl than furst one." << endl;
//		return -1;
//	}


	if ( !(argc-1 == optind) ){
		cerr << "No image file was specified.\n" << endl;
		help(argv[0]);
		return -1;
	}
	imagefile = argv[optind];

	{
		long fsize = get_filesize( imagefile );
		if ( fsize == -1 ){
			cerr << "File Error" << endl;
			return -1;
		}
		if ( fsize > MAX_FILESIZE ){
			cerr << "File size(" << fsize << ") over (MAX: " << MAX_FILESIZE << " byte)" << endl;
			return -1;
		}
	}

	// -- init ---
	int histogram[ NUM_OF_BIN ];
	float chromes[ NUM_OF_BIN ];

	for ( int i=0; i<NUM_OF_BIN; i++ ){
		histogram[i] = 0;
		chromes[i]   = 0;
	}

	// -- exec to calc ---
	ret = calcHistogram(imagefile, histogram, chromes);

	if (ret < 0) {
		cerr << "Histgram calculation failed." << endl;
		return -1;
	}

	// -- Calculation ---
	int peakbin = -1;
	int peakval = 0;

	int peakbin_c = -1;
	int peakval_c = 0;

	int peakbin_c2 = -1;
	int peakval_c2 = 0;

	for ( int i = 0; i<NUM_OF_BIN; i++ )
	{
		int h   = histogram[i];
		float c = chromes[i];

		// record most used color
		// memo: using colmunar moder, make threshould parameter rather big.
		// (ex. 0.6 or so)
		// conic model, it is better to be smaller than it. (ex. 0.3 or so)
		if ( h > peakval ){
			peakbin = i;
			peakval = h;
		}

		if ( peakonly ) continue;

		// and one limitated by -c
		if ( c >= crate ){
			if ( h > peakval_c ){
				peakbin_c = i;
				peakval_c = h;
			}
		}

		// and least chrome 0.2
		if ( c >= crate2 ){
			if ( h > peakval_c2 ){
				peakbin_c2 = i;
				peakval_c2 = h;
			}
		}
	}

	// -- output result ---
	int retbin = 0;
	int retrgb[3];

	if ( peakbin_c != -1 ){
		retbin = peakbin_c;
	} else {
		if ( peakbin_c2 != -1 ){
			retbin = peakbin_c2;
		} else {
			retbin = peakbin;
		}
	}
	bin2rgb( retbin, retrgb);

	if ( sstout ){
		// CSS format
		if ( COLDEPTH > 4 ){
			printf("#%02x%02x%02x", retrgb[0], retrgb[1], retrgb[2]);	// #a1b2c3
		} else {
			printf("#%x%x%x", retrgb[0], retrgb[1], retrgb[2]);	// #abc
		}
	} else {
		if ( hexout ){
			printf("%x %x %x", retrgb[0], retrgb[1], retrgb[2]);
		} else {
			printf("%d %d %d", retrgb[0], retrgb[1], retrgb[2]);
		}
	}
}


static void help(char *path)
{
	printf("USAGE:\n"
		"   %s [-p] [-x|-s] [-c F_CHROMA] [-n S_CHROMA] IMAGEFILE\n"
		"\t-p          ... Simply, most used color picking\n"
		"\t-s          ... Output CSS format\n"
		"\t-x          ... Output with hexadecimals\n"
		"\t-c F_CHROMA ... 0<n<1.0. First pickup threshold(Def 0.5)\n"
		"\t-n S_CHROMA ... 0<n<1.0. Second pickup threshold(Def 0.2)\n"
		"\tIMAGEFILE   ... Image file\n"
		,path
	);
}
