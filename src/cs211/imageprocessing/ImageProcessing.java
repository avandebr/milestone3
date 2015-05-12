package cs211.imageprocessing;

import static cs211.imageprocessing.QuadGraph.isConvex;
import static cs211.imageprocessing.QuadGraph.nonFlatQuad;
import static cs211.imageprocessing.QuadGraph.validArea;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import processing.core.PApplet;
import processing.core.PImage;
import processing.core.PVector;
import processing.video.Capture;

@SuppressWarnings("serial")
public class ImageProcessing extends PApplet {

	Capture cam;
	PImage img;
	PImage houghImg;

	float discretizationStepsPhi; 
	float discretizationStepsR;
	int phiDim;

	float[] tabSin;
	float[] tabCos;

	public void setup() { 

		size(1200, 300);
		
		discretizationStepsPhi = 0.06f; 
		discretizationStepsR = 2.5f;
		phiDim = round(PI / discretizationStepsPhi);
		tabSin = new float[phiDim];
		tabCos = new float[phiDim];
		float ang = 0;
		float inverseR = 1.f / discretizationStepsR;
		for (int accPhi = 0; accPhi < phiDim; ang += discretizationStepsPhi, accPhi++) {
			// we can also pre-multiply by (1/discretizationStepsR) since we need it in the Hough loop 
			tabSin[accPhi] = (float) (sin(ang) * inverseR);
			tabCos[accPhi] = (float) (cos(ang) * inverseR);
		}

		String[] cameras = Capture.list(); 
		if (cameras.length == 0) {
			println("There are no cameras available for capture.");
			exit(); 
		} 
		else {
			println("Available cameras:");
			for (int i = 0; i < cameras.length; i++) {
				println(cameras[i]);
			}
			cam = new Capture(this, cameras[0]);
			cam.start();
			img = loadImage("board1.jpg");
			img.resize(width/3, height);
		}
		noLoop();  // no interactive behaviour: draw() will be called only once.
	}

	public void draw() {

		/*if (cam.available() == true) {
			cam.read();
		}
		img = cam.get();*/
		
		image(img, 0, 0);
		getIntersections(pipeline(img));
		image((houghImg), width/3, 0);
		image(sobel(intenistyThresholding(convolute(deleteNoise(img)))), 2*width/3, 0);
	}
	
	public ArrayList<PVector> pipeline(PImage img){
		return hough(sobel(intenistyThresholding(convolute(deleteNoise(img)))), 4);
	}

	public float mySin(float phi){
		int accPhi = round(phi/discretizationStepsPhi);
		return discretizationStepsR * ((accPhi < phiDim) ?
				tabSin[accPhi] : tabSin[phiDim-1]);
	}
	public float myCos(float phi){
		int accPhi = round(phi/discretizationStepsPhi);
		return discretizationStepsR * ((accPhi < phiDim) ?
				tabCos[accPhi] : tabCos[phiDim-1]);
	}

	public PImage deleteNoise(PImage img){

		PImage result = createImage(img.width, img.height, RGB); 

		final double minHue = 100;
		final double maxHue = 135;

		final double minBrightness = 30;
		final double maxBrightness = 180;

		final double minSaturation = 0;
		final double maxSaturation = 100;

		for(int i = 0; i < img.width * img.height; i++) {
			if((hue(img.pixels[i]) < maxHue && hue(img.pixels[i]) > minHue)
					&& (brightness(img.pixels[i]) < maxBrightness && brightness(img.pixels[i]) > minBrightness)
					&& !(saturation(img.pixels[i]) < maxSaturation && saturation(img.pixels[i]) > minSaturation)) {
				result.pixels[i] = color(255);
			}
			else result.pixels[i] = color(0);
		}
		return result;
	}

	public PImage convolute(PImage img) {

		float[][] kernel = {{9, 12, 9},
							{12, 15, 12},
							{9, 12, 9}};	

		float weight = 99.f;
		PImage result = createImage(img.width, img.height, ALPHA);
		final int N = kernel.length;
		int sum;

		for (int x = N/2; x < img.width - N/2; x++) {
			for (int y = N/2; y < img.height - N/2; y++) {
				sum = 0;
				for (int i = 0; i < N; i++){
					for (int j = 0; j < N; j++){
						sum += brightness(img.pixels[x - N/2 + i + (y - N/2 + j) * img.width]) * kernel[i][j];
					}
				}
				result.pixels[x + y * img.width] = color(sum / weight);
			}
		}
		return result;
	}
	
	public PImage intenistyThresholding(PImage img){
		
		PImage result = createImage(img.width, img.height, RGB);
		for(int i = 0; i < img.width * img.height; i++) {
			result.pixels[i] = color(brightness(img.pixels[i]) > 1 ? 255 : 0);
		}
		return result;
	}

	public PImage sobel(PImage img){

		float[][] hKernel = {{0, 1, 0},
				{0, 0, 0},
				{0, -1, 0}};

		float[][] vKernel = {{0, 0, 0},
				{1, 0, -1},
				{0, 0, 0}};

		PImage result = createImage(img.width, img.height, ALPHA);

		for(int i = 0; i < img.width*img.height; i++){
			result.pixels[i] = color(0);
		}

		float max = 0;
		float[] buffer = new float[img.width * img.height];

		int sum_h = 0, sum_v = 0, sum = 0;
		final int N = hKernel.length;

		for (int x = N/2; x < img.width - N/2; x++) {
			for (int y = N/2; y < img.height - N/2; y++) {
				sum_h = 0;
				sum_v = 0;
				sum = 0;
				for (int i = 0; i < N; i++){
					for (int j = 0; j < N; j++){
						sum_h += img.pixels[x - N/2 + i + (y - N/2 + j) * img.width] * hKernel[i][j];
						sum_v += img.pixels[x - N/2 + i + (y - N/2 + j) * img.width] * vKernel[i][j];
					}
				}
				sum = round(sqrt(pow(sum_h, 2) + pow(sum_v, 2)));
				if(sum > max){
					max = sum;
				}
				buffer[x + y*img.width] = sum;
			}
		}

		for(int y = 2; y < img.height; y++){
			for(int x = 2; x < img.width; x++){
				if(buffer[y*img.width + x] > round(max*0.3f)){
					result.pixels[y*img.width + x] = color(255);
				}
				else {
					result.pixels[y*img.width + x] = color(0);
				}
			}
		}
		return result;
	}

	public ArrayList<PVector> hough(PImage edgeImg, int nLines) {

		int rDim = round(((edgeImg.width + edgeImg.height) * 2 + 1) / discretizationStepsR);
		int[] accumulator = new int[(phiDim + 2) * (rDim + 2)];

		float r = 0;
		int accPhi = 0;
		int accR = 0;
		int idx = 0;

		// our accumulator (with a 1 pix margin around)
		// Fill the accumulator: on edge points (ie, white pixels of the edge image), 
		//store all possible (r, phi) pairs describing lines going through the point.
		for (int y = 0; y < edgeImg.height; y++) {
			for (int x = 0; x < edgeImg.width; x++) {
				// Are we on an edge?
				if (brightness(edgeImg.pixels[y * edgeImg.width + x]) != 0) {
					// ...determine here all the lines (r, phi) passing through
					// pixel (x,y), convert (r,phi) to coordinates in the
					// accumulator, and increment accordingly the accumulator.
					for (float phi = 0; phi < PI; phi += discretizationStepsPhi) {
						r = x * myCos(phi) + y * mySin(phi);
						accPhi = round(phi / discretizationStepsPhi);
						accR = round(r / discretizationStepsR + (rDim - 1) * 0.5f);
						idx = (accPhi + 1) * (rDim + 2) + accR + 1;
						accumulator[idx]++;
					}
				} 
			}
		}

		houghImg = createImage(rDim + 2, phiDim + 2, ALPHA);
		for (int i = 0; i < accumulator.length; i++) {
			houghImg.pixels[i] = color(min(255, accumulator[i]));
		}
		houghImg.updatePixels();
		houghImg.resize(width/3, height);

		ArrayList<Integer> bestCandidates = new ArrayList<>();
		ArrayList<PVector> bestVectors = new ArrayList<>();
		// size of the region we search for a local maximum
		int neighbourhood = 10;
		int minVotes = 100;
		// only search around lines with more that this amount of votes // (to be adapted to your image)
		for (accR = 0; accR < rDim; accR++) {
			for (accPhi = 0; accPhi < phiDim; accPhi++) {
				// compute current index in the accumulator
				idx = (accPhi + 1) * (rDim + 2) + accR + 1; 
				if (accumulator[idx] > minVotes) {
					boolean bestCandidate = true;
					// iterate over the neighbourhood
					for(int dPhi=-neighbourhood/2; dPhi < neighbourhood/2+1; dPhi++) { 
						// check we are not outside the image
						if(accPhi+dPhi < 0 || accPhi+dPhi >= phiDim) continue; 
						for(int dR=-neighbourhood/2; dR < neighbourhood/2 +1; dR++) {
							// check we are not outside the image
							if(accR+dR < 0 || accR+dR >= rDim) continue;
							int neighbourIdx = (accPhi + dPhi + 1) * (rDim + 2) + accR + dR + 1;
							if(accumulator[idx] < accumulator[neighbourIdx]) { 
								// the current idx is not a local maximum! bestCandidate=false;
								break;
							} 
						}
						if(!bestCandidate) break; 
					}
					if(bestCandidate) {
						// the current idx *is* a local maximum
						bestCandidates.add(idx);
					}
				} 
			}
		}

		Collections.sort(bestCandidates, new HoughComparator(accumulator));
		for(int i = 0; i < nLines && i < bestCandidates.size(); i++){

			idx = bestCandidates.get(i);
			// first, compute back the (r, phi) polar coordinates:

			accPhi = round(idx / (rDim + 2)) - 1;
			accR = idx - (accPhi + 1) * (rDim + 2) - 1;
			r = (accR - (rDim - 1) * 0.5f) * discretizationStepsR; 
			float phi = accPhi * discretizationStepsPhi;
			bestVectors.add(new PVector(r, phi));
			// Cartesian equation of a line: y = ax + b
			// in polar, y = (-cos(phi)/sin(phi))x + (r/sin(phi))
			// => y = 0 : x = r / cos(phi)
			// => x = 0 : y = r / sin(phi)
			// compute the intersection of this line with the 4 borders of the image
			int x0 = 0;
			int y0 = round(r / mySin(phi));
			int x1 = round(r / myCos(phi));
			int y1 = 0;
			int x2 = edgeImg.width;
			int y2 = round(-myCos(phi) / mySin(phi) * x2 + r / mySin(phi)); 
			int y3 = edgeImg.width;
			int x3 = round(-(y3 - r / mySin(phi)) * (mySin(phi) / myCos(phi)));
			// Finally, plot the lines
			stroke(204,102,0); 
			if (y0 > 0) {
				if (x1 > 0)
					line(x0, y0, x1, y1);
				else if (y2 > 0)
					line(x0, y0, x2, y2);
				else
					line(x0, y0, x3, y3);
			}
			else {
				if (x1 > 0) {
					if (y2 > 0)
						line(x1, y1, x2, y2); 
					else
						line(x1, y1, x3, y3);
				}
				else
					line(x2, y2, x3, y3);
			}
		}
		return bestVectors;
	}
	
	public PImage drawQuads(PImage img){
		ArrayList<PVector> lines = pipeline(img);
		QuadGraph quads = new QuadGraph();
		quads.build(lines, img.width, img.height);
		for (int[] quad : quads.findCycles()) {
	        PVector l1 = lines.get(quad[0]);
	        PVector l2 = lines.get(quad[1]);
	        PVector l3 = lines.get(quad[2]);
	        PVector l4 = lines.get(quad[3]);
	        // (intersection() is a simplified version of the
	        // intersections() method you wrote last week, that simply
	        // return the coordinates of the intersection between 2 lines) 
	        PVector c12 = intersection(l1, l2);
	        PVector c23 = intersection(l2, l3);
	        PVector c34 = intersection(l3, l4);
	        PVector c41 = intersection(l4, l1);

	        final float windowArea = img.width*img.height;
	        if(isConvex(c12, c23, c34, c41) 
	        		&& validArea(c12, c23, c34, c41, 3*windowArea/4, windowArea/4) && nonFlatQuad(c12, c23, c34, c41)){
	        // Choose a random, semi-transparent colour
	        	Random random = new Random(); 
	        	fill(color(min(255, random.nextInt(300)),
	        			min(255, random.nextInt(300)),
	        			min(255, random.nextInt(300)), 50));
	        	quad(c12.x,c12.y,c23.x,c23.y,c34.x,c34.y,c41.x,c41.y);
			}
	    }
		return img;
	}
	
	public PVector intersection(PVector line1, PVector line2) { 
		
		float r1 = line1.x, r2 = line2.x;
		float phi1 = line1.y, phi2 = line2.y;
		float d = myCos(phi2)*mySin(phi1) - myCos(phi1)*mySin(phi2);
		float x = (r2*mySin(phi1) - r1*mySin(phi2)) / d;
		float y = (-r2*myCos(phi1) + r1*myCos(phi2)) / d;
			
		return new PVector(x, y); 
	}

	public ArrayList<PVector> getIntersections(List<PVector> lines) { 
		ArrayList<PVector> intersections = new ArrayList<PVector>(); 
		for (int i = 0; i < lines.size() - 1; i++) {
			PVector line1 = lines.get(i);
			for (int j = i + 1; j < lines.size(); j++) {
				PVector line2 = lines.get(j);
				float r1 = line1.x, r2 = line2.x;
				float phi1 = line1.y, phi2 = line2.y;
				float d = myCos(phi2)*mySin(phi1) - myCos(phi1)*mySin(phi2);
				float x = (r2*mySin(phi1) - r1*mySin(phi2)) / d;
				float y = (-r2*myCos(phi1) + r1*myCos(phi2)) / d;
				intersections.add(new PVector(x, y));
				fill(255, 128, 0);
				ellipse(x, y, 10, 10);
			}
		}
		return intersections; 
	}

	class HoughComparator implements Comparator<Integer> { 
		int[] accumulator;
		public HoughComparator(int[] accumulator) {
			this.accumulator = accumulator; 
		}
		@Override
		public int compare(Integer l1, Integer l2) { 
			return (accumulator[l1] > accumulator[l2] 
				|| (accumulator[l1] == accumulator[l2] && l1 < l2)) ?
					-1 : 1;
		} 
	}

}
