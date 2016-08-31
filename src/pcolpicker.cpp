// =====================================================================
//  pcolpicker - PricolorPicker
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//    get primary color from image file.
//
//    This program picks up primary color of image file.
//    Target image is a nail photo, so this discards skin color.
//
//    [Logic]
//      1. The program loads an images from file or STDIN.
//      2. Clip image (reason that almost objects (fingers or nails)
//         are fourcused in center of image).
//      3. Resize image bit smaller for using less memory.
//      4. Make image posterization.
//      5. Convert it from RGB to HSV.
//      6. Make histgram with high satuation.
//
//    ---
//    Developed by:
//        T.Onodera 
//    Version:
//        2.0.0
//    Cf.:
//        http://aidiary.hatenablog.com/entry/20091003/1254574041
//
// =====================================================================

#include <cstdio>
#include <iostream>
#include <utility>
#include <stdlib.h>
#include <getopt.h>
#include <fcntl.h>
#include <unistd.h> 
#include <vector> 
#include <sys/types.h>
#include <sys/stat.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>


#define DEBUG	0
#define VERSION "2.0.1"

// STDIN uging without filename
#define USE_STDIN 1
#define READ_BUFFER_SIZE     2048
// Filesize limitation
#define MAX_FILESIZE    ((unsigned long)1000000000)

// Default values
#define DEF_HBINS	2
#define DEF_SBINS	4
#define DEF_HRANGE0  -3		// -6'/360'
#define DEF_HRANGE1  24		// 48'/360'	Covering pR,R,yR of PCCS
#define DEF_SLEVEL1  200	// 78%/100%
#define DEF_CLIPRATIO  0.1
#define DEF_IGNORE_MONO_LEVEL 40 // about 15% of chromatic

#define DEF_RESIZE_WIDTH  200
#define DEF_RESIZE_HEIGHT 200

#define DEF_NORMALIZE_KERNELSIZE 0
#define DEF_BITREDUCE_POSTERIZE  2


#define DEF_WHITING_RATIO 0.0	// output color with bit whiting


using namespace std;
using namespace cv;


void getPriColorHSV(Mat &, Mat &, Mat &, int, int, int *, int *, int);
long get_filesize(const char *);
static void show_help(char *);
static void show_version(char *);


/**
 * Calculate and analyse color of image and return it by HSV
 */
void getPriColorHSV(Mat& image, Mat& rhsv, int hbins, int sbins, int* hranges, int* sranges, int peakonly)
{
	// bin counter
	int elemsize[] = { 180>>hbins, 256>>sbins, 256>>sbins };
	SparseMat counter(3, elemsize, CV_16UC1);	// max 65536 count

	// Convert image to HSV data.
	// H {0...180}, S {0..255}, V{0..255}
	Mat hsv;
	cvtColor( image, hsv, CV_BGR2HSV );

	// pixel counting
	int hrange0 = hranges[0];
	int hrange1 = hranges[1];
	int srange0 = sranges[0];
	int srange1 = sranges[1];

	int totalcount = 0;
	for ( int y = 0; y < hsv.rows; y++ ){
		for ( int x = 0; x < hsv.cols; x++ ){
			Vec3b vec = hsv.at<Vec3b>(y,x);

			int hue = vec[0];
			int sat = vec[1];
			int val = vec[2];

			if ( !peakonly ){
				if ( sat < DEF_IGNORE_MONO_LEVEL ) continue;     // ignore monotone
				// skip when color is out of range
				if ( srange0 <= sat && sat <= srange1 ){
					if ( hrange0 < 0 ){
						if ( hrange0+180 <= hue && hue <= 180 || 0 <= hue && hue <= hrange1 ){
							continue;
						};
					} else {
						if ( hrange0 <= hue && hue <= hrange1 ){
							continue;
						}
					}
				}
			}

			hue = (hue>>hbins)<<hbins;
			sat = (sat>>sbins)<<sbins;
			val = (val>>sbins)<<sbins;

			counter.ref<int>(hue, sat, val)++;
			totalcount++;
		}
	}

	// --- find most used color ---
	rhsv = Mat(1,1,CV_8UC3);

	if ( totalcount ){
		SparseMatIterator it     = counter.begin(),
		                  it_end = counter.end();
		Vec3b maxidx(0,0,0);
		int   maxcnt = 0;
		for (; it != it_end; ++it ){
			const SparseMat::Node* n = it.node();
			int count = it.value<int>();
	//		cout << "cnt: " << it.value<int>() << endl;

			if ( count > maxcnt ){
				int hue = n->idx[0];
				int sat = n->idx[1];
				int val = n->idx[2];
#if DEBUG
				cerr <<"hue:" << hue << ", sat:" << sat << ", val:" << val << "  -> cnt: " << count <<endl;
#endif
				maxidx = Vec3b(hue, sat, val);
				maxcnt = count;
			}
		}
#if DEBUG
		cerr << "MaxCnt: " << maxcnt << endl;
		cerr << "MaxIdx: " << maxidx << endl;
#endif

		rhsv.at<Vec3b>(0,0) = maxidx;

	} else {
		// return black
		rhsv.at<Vec3b>(0,0) = Vec3b(0, 0, 0);
	}
}


/**
 * get byte num of file size
 */

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
	int peakonly = 0;
	int out_dec  = 0;
	int out_css  = 0;
	int out_hsv  = 0;
	int hbins    = DEF_HBINS;
	int sbins    = DEF_SBINS;
	int hrange0  = DEF_HRANGE0;
	int hrange1  = DEF_HRANGE1;
	int slevel1  = DEF_SLEVEL1;
	int do_normal= DEF_NORMALIZE_KERNELSIZE;
	int do_reduce= DEF_BITREDUCE_POSTERIZE;
	float ratioclip = DEF_CLIPRATIO;
	float whiting= DEF_WHITING_RATIO;


	Mat             image;
	char            *imagefile;
	vector<uchar>   imgbuff;	//buffer for coding
	char            readbuf[READ_BUFFER_SIZE];


	int arg;
	while( ( arg = getopt(argc, argv, "hvpdxmn:b:s:a:z:c:l:w:")) != -1 )
	{
		switch(arg){
			default:
			case 'h':
				show_help(argv[0]);
				return 0;
				break;

			case 'v':
				show_version(argv[0]);
				return 0;
				break;

			case '?':
				cerr << "Unknown option." << endl;
				return -1;
				break;

			case 'p':
				peakonly = 1;
				break;

			case 'd':	// decimal output
				out_dec = 1;
				break;

			case 'x':	// hex output
				out_css = 1;
				break;

			case 'm':	// hsv out
				out_hsv = 1;
				break;

			case 'b':
				hbins = (int)atoi(optarg);
				if ( hbins < 0 || hbins > 5 ){
					cerr << "Param -b error" << endl;
					return -1;
				}
				break;

			case 's':
				sbins = (int)atoi(optarg);
				if ( sbins < 0 || sbins > 7 ){
					cerr << "Param -s error" << endl;
					return -1;
				}
				break;

			case 'a':
				hrange0 = (int)atoi(optarg);
				if ( hrange0 < -179 || hrange0 > 180 ){
					cerr << "Param -a error" << endl;
					return -1;
				}
				break;

			case 'z':
				hrange1 = (int)atoi(optarg);
				if ( hrange1 < -179 || hrange1 > 180 ){
					cerr << "Param -z error" << endl;
					return -1;
				}
				break;

			case 'c':
				slevel1 = (int)atoi(optarg);
				if ( slevel1 < 0 || slevel1 > 255 ){
					cerr << "Param -c error" << endl;
					return -1;
				}
				break;

			case 'l':
				ratioclip = (float)atof(optarg);
				if ( ratioclip < 0 || ratioclip > 0.9 ){
					cerr << "Param -l error" << endl;
					return -1;
				}
				break;

			case 'w':
				whiting = (float)atof(optarg);
				if ( whiting < 0 || whiting > 2.0 ){
					cerr << "Param -w error" << endl;
					return -1;
				}
				break;

			case 'n':	// Execute normarization with medianBlur
				do_normal = (int)atoi(optarg);
				if ( do_normal < 1 || do_normal > 9 || (do_normal%2 == 0) ){
					cerr << "Param -n error" << endl;
					return -1;
				}
				break;
		}
	}

	if ( hrange0 > hrange1 ){
		swap(hrange0, hrange1);
	}

#if !USE_STDIN
	if ( !(argc-1 == optind) ){
		cerr << "No image file was specified.\n" << endl;
		show_help(argv[0]);
		return -1;
	}
	imagefile = argv[optind];
#else
	if ( !(argc-1 == optind) ){
		imagefile = NULL;
	} else {
		imagefile = argv[optind];
	}
#endif


	// ----- load image -----
	long size=0, fsize = 0;

	if ( imagefile == NULL ){

		// from STDIN
		while ( (size = read(fileno(stdin), readbuf, READ_BUFFER_SIZE)) > 0 ){
			imgbuff.reserve(size);
			fsize += size;
			if ( fsize > MAX_FILESIZE ){
				cerr << "File size(" << fsize << ") over (MAX: " << MAX_FILESIZE << " byte)" << endl;
				return -1;
			}
			for (int i=0; i<size; i++) imgbuff.push_back(readbuf[i]);
		}
		image = imdecode(Mat(imgbuff), CV_LOAD_IMAGE_COLOR);

	} else {

		// from strage
		fsize = get_filesize( imagefile );
		if ( fsize == -1 ){
			cerr << "File Error" << endl;
			return -1;
		}
		if ( fsize > MAX_FILESIZE ){
			cerr << "File size(" << fsize << ") over (MAX: " << MAX_FILESIZE << " byte)" << endl;
			return -1;
		}
		image = imread( imagefile, CV_LOAD_IMAGE_COLOR );
	}

	if ( image.empty() ){
		cerr << "Image data error" << endl;
		return -1;
	}


	// ----- Modify image data to calculate -----
	int w0,h0,cw,ch;
	w0 = image.cols;
	h0 = image.rows;
	cw = (int)(w0 * ratioclip);
	ch = (int)(h0 * ratioclip);
	Rect rect = Rect(cw/2, ch/2, w0-cw, h0-ch );
#if DEBUG
	cerr << "Clipping: " << rect <<endl;
#endif

	Mat modimg;
	resize(
		image( rect ),
		modimg,
		Size( DEF_RESIZE_WIDTH, DEF_RESIZE_HEIGHT ),
		INTER_LINEAR
	);
	image = modimg;

	if ( do_normal ){
		medianBlur(image, modimg, do_normal);
		image = modimg;
	}

#if DEBUG
    imwrite("/tmp/testout.png", image);
#endif

	// ----- Histgram Calculation -----
	Mat rhsv = Mat(1,1,CV_8UC3);
	Mat rgb = Mat(1,1,CV_8UC3);
	int hranges[] = { hrange0, hrange1 };
	int sranges[] = { 0, slevel1 };
	getPriColorHSV( image, rhsv, hbins, sbins, hranges, sranges, peakonly );

#if DEBUG
	cerr << "Returned HSV: " << rhsv <<endl;
#endif

	if ( whiting > 0.0 ){
		Vec3b hsv = rhsv.at<Vec3b>(0,0);
		int sat = (float)hsv[2] / whiting;
		rhsv.at<Vec3b>(0,0) = Vec3b( hsv[0], sat, hsv[2] );
#if DEBUG
		cerr << "Whited HSV: " << rhsv <<endl;
#endif
	}

	cvtColor( rhsv, rgb, CV_HSV2RGB );
#if DEBUG
		cerr << "RGB to get: " << rgb <<endl;
#endif

	if ( out_hsv ){
		Vec3b hsv = rhsv.at<Vec3b>(0,0);
		printf("%d %d%% %d%%", hsv[0]*360/180, hsv[1]*100/256, hsv[2]*100/256);	// H S V
	} else {
		Vec3b ret = rgb.at<Vec3b>(0,0);

		if ( out_css ){
			// CSS format
			printf("#%02x%02x%02x", ret[0], ret[1], ret[2]);	// #a1b2c3
		} else {
			if ( out_dec ){
				printf("%d %d %d", ret[0], ret[1], ret[2]);
			} else {
				printf("%02x%02x%02x", ret[0], ret[1], ret[2]);	// #aabbcc
			}
		}
	}
}


static void show_help(char *path)
{
	printf("USAGE:\n"
		"   %s [-p] [-d|-x] [-b SIZE] [-c SIZE] [-a DEGREE] [-z DEGREE] [-c LEVEL] [IMAGEFILE]\n"
		"\t-p          ... Most used color output picking without range limits\n"
		"\t-m          ... HSV output\n"
		"\t-d          ... Output with decimals\n"
		"\t-x          ... Output with stylesheet format\n"
		"\t-b SIZE     ... 0~5. Bit shift amount of Hue (Default %d)\n"
		"\t-s SIZE     ... 0~7. Bit shift amount of Saturation (Default %d)\n"

		"\t-a DEGREE   ... -179~180. Start degree of exception lange of hue (Default %d)\n"
		"\t-z DEGREE   ... -179~180. End digree of exception lange of hue  (Default %d)\n"
		"\t-c LEVEL    ... 0~255. upper level of chrome to ignore (Default %d)\n"

		"\t-l CLIP     ... 0~0.9. Ratio of cliping (Default %.1f)\n"
		"\t-n:         ... 1,3,5,7 or 9. Normalization level. Omiting or 0 ignores. (Default %d)\n"
		"\t-w:         ... 0.0~2.0. Multipul number to white. (Default %.1f)\n"
		"\tIMAGEFILE   ... Image file. Omitting means from STDIN\n"
		"\t-v          ... Print version\n"
		,path, DEF_HBINS, DEF_SBINS, DEF_HRANGE0, DEF_HRANGE1,DEF_SLEVEL1,
		DEF_CLIPRATIO, DEF_NORMALIZE_KERNELSIZE,DEF_WHITING_RATIO
	);
}

static void show_version(char *path)
{
	printf("PriColor Picker - version %s\n", VERSION);
}

