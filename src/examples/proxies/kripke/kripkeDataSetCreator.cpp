#include <stdio.h>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <fstream>
#include <set>
#include <stdlib.h>
#include <eavlDataSet.h>
#include <eavlArray.h>
#include <eavlCellSetExplicit.h>
#include <eavlCellSetAllStructured.h>
#include <eavlCoordinates.h>
#include <eavlLogicalStructureRegular.h>
#include <eavlVTKExporter.h>
using namespace std;

double minExtents[3];
double maxExtents[3];
double minTime, maxTime;
double aveTimeDiff = 0;
int totalTimes = 0;
set<double> xAxis;
set<double> yAxis;
set<double> zAxis;
struct Snapshot
{
  double requestSize;
  double queueSize;
  double time;
};

struct NodeData
{
  double extents[2][3];
  int cellIndex;     //index into the dataset nodal field array
  int mpi_rank;
  vector<Snapshot> snapshots;
};

NodeData* readFile(string filename)
{
  ifstream datafile;
  datafile.open(filename.c_str(), ios::in | ios::binary);
  if(!datafile.is_open())
  {
    cerr<<"Could not open file "<<filename<<endl;
    exit(1);
  }
  
  NodeData *data = new NodeData();
  
  datafile.read((char*)&data->mpi_rank, sizeof(int));
  datafile.read((char*)&data->extents[0][0], 3 * sizeof(double));
  datafile.read((char*)&data->extents[1][0], 3 * sizeof(double));
  #pragma omp critical
  {
    minExtents[0] = min(minExtents[0], data->extents[0][0]);
    minExtents[1] = min(minExtents[1], data->extents[0][1]);
    minExtents[2] = min(minExtents[2], data->extents[0][2]);

    
    maxExtents[0] = max(maxExtents[0], data->extents[1][0]);
    maxExtents[1] = max(maxExtents[1], data->extents[1][1]);
    maxExtents[2] = max(maxExtents[2], data->extents[1][2]);
    //
    // Do some rounding so we get the correct values
    //
    float places = 10000;
    float value = roundf(data->extents[0][0] * places) / places;  
    xAxis.insert(value);
    value = roundf(data->extents[1][0] * places) / places;
    xAxis.insert(value);
    value = roundf(data->extents[0][1] * places) / places;
    yAxis.insert(value);
    value = roundf(data->extents[1][1] * places) / places;
    yAxis.insert(value);
    value = roundf(data->extents[0][2] * places) / places;
    zAxis.insert(value);
    value = roundf(data->extents[1][2] * places) / places;
    zAxis.insert(value);
  }
  int count;
  datafile.read((char*)&count, sizeof(int));
  if(count < 1)
  {
    cerr<<"Rank "<<data->mpi_rank<<" had no data for extent :"<<endl;
    cerr<<data->extents[0][0]<<" "<<data->extents[0][1]<<" "<<data->extents[0][2]<<"<-> "
        <<data->extents[1][0]<<" "<<data->extents[1][1]<<" "<<data->extents[1][2]<<endl;
  }

  for(int i = 0; i < count; i++)
  {
    Snapshot snapshot;
    int rs, qs;
    long long papiCounter;
    datafile.read((char*) &(snapshot.time), sizeof(double));         
		datafile.read((char*) &(rs), sizeof(int));
		datafile.read((char*) &(qs), sizeof(int));
		datafile.read((char*) &(papiCounter), sizeof(long long));
		//cout<<"Flops "<<papiCounter<<" ";
		//cout<<snapshot.time<<" "<<snapshot.requestSize<<" "<<snapshot.queueSize<<endl;
		snapshot.requestSize = rs;
		snapshot.queueSize = qs;
		minTime = min(minTime, snapshot.time);
		maxTime = max(maxTime, snapshot.time);
		data->snapshots.push_back(snapshot);
		if(i != 0)
		{
		  totalTimes++;
		  aveTimeDiff += snapshot.time - data->snapshots.at(i - 1).time;
		}
  }
  datafile.close();
  return data;
}

void createDataSets(NodeData **data, int numNodes)
{
  //
  // We need to figure out the grid and give each Node 
  // a cell index
  //
  int nx = xAxis.size();
  int ny = yAxis.size();
  int nz = zAxis.size();
  
  //Create the coordinate axis
  double *xCoords = new double[nx];
  double *yCoords = new double[ny];
  double *zCoords = new double[nz]; 
  int idx = 0;
  for (set<double>::iterator it=xAxis.begin(); it!=xAxis.end(); ++it)
  {
    xCoords[idx] = *it;
    idx++; 
  }
	idx = 0;
	for (set<double>::iterator it=yAxis.begin(); it!=yAxis.end(); ++it)
  {
    yCoords[idx] = *it;
    idx++; 
  }
  idx = 0;
	for (set<double>::iterator it=zAxis.begin(); it!=zAxis.end(); ++it)
  {
    zCoords[idx] = *it;
    idx++; 
  }
  cout<<"\nx axis"<<endl;
  for(int i = 0; i < nx; i++) cout<<" "<<xCoords[i];
  cout<<"\ny axis"<<endl;
  for(int i = 0; i < ny; i++) cout<<" "<<yCoords[i];
  cout<<"\nz axis"<<endl;
  for(int i = 0; i < nz; i++) cout<<" "<<zCoords[i];
  cout<<endl;
  for(int i = 0; i < numNodes; i++)
  {
    // get the botom left corner of the cell
    
    double x = data[i]->extents[0][0];
    double y = data[i]->extents[0][1];
    double z = data[i]->extents[0][2];
    int xIdx = -1;
    for(int c = 0; c < nx; c++)
    {
      if(xCoords[c] == x) xIdx = c; 
    }
    if(xIdx == -1)
    {
      cerr<<"Could not locate x coord "<<x<<endl;
    }
    int yIdx = -1;
    for(int c = 0; c < ny; c++)
    {
      if(yCoords[c] == y) yIdx = c; 
    }
    if(yIdx == -1)
    {
      cerr<<"Could not locate y coord "<<x<<endl;
    }
    int zIdx = -1;
    for(int c = 0; c < nz; c++)
    {
      if(zCoords[c] == z) zIdx = c; 
    }
    if(zIdx == -1)
    {
      cerr<<"Could not locate z coord "<<x<<endl;
    }
    //cout<<"Node "<<i<<" at "<<xIdx<<","<<yIdx<<","<<zIdx<<endl;
    
    int cellIdx = zIdx*(nx-1)*(ny-1)+yIdx*(nx-1)+xIdx;
    //cout<<"Cell index "<<cellIdx<<endl;
    data[i]->cellIndex = cellIdx;
   // cout<<"Bottom left coords "<<xCoords[xIdx]<<" "<<yCoords[yIdx]<<" "<<zCoords[zIdx]<<endl;
    
  } 
  
  //
  //  We now have everything we need to create the data set
  //
  
  int nels = (nx - 1) *(ny - 1) *(nz - 1);
  int npts = nx * ny * nz;
  eavlDataSet *dataset = new eavlDataSet();
  eavlRegularStructure reg;
  
  reg.SetNodeDimension3D(nx, ny, nz);
  eavlLogicalStructure *log =
            new eavlLogicalStructureRegular(reg.dimension, reg);
  // Create the coordinate axes.
  eavlFloatArray *x, *y, *z;
  x = new eavlFloatArray("x", 1, nx);
    for (int i = 0; i < nx; i++) 
      x->SetValue(i, xCoords[i]);
  y = new eavlFloatArray("y", 1, ny);
    for (int i = 0; i < ny; i++) 
      y->SetValue(i, yCoords[i]);
  z = new eavlFloatArray("z", 1, nz);
    for (int i = 0; i < nz; i++) 
      z->SetValue(i, zCoords[i]);
  dataset->AddField(new eavlField(1, x, eavlField::ASSOC_LOGICALDIM, 0));
  dataset->AddField(new eavlField(1, y, eavlField::ASSOC_LOGICALDIM, 1));
  dataset->AddField(new eavlField(1, z, eavlField::ASSOC_LOGICALDIM, 2));
  eavlCoordinates *coords = new eavlCoordinatesCartesian(log,
                                eavlCoordinatesCartesian::X,
                                eavlCoordinatesCartesian::Y,
                                eavlCoordinatesCartesian::Z);
  coords->SetAxis(0, new eavlCoordinateAxisField("x"));
  coords->SetAxis(1, new eavlCoordinateAxisField("y"));
  coords->SetAxis(2, new eavlCoordinateAxisField("z"));                              
  dataset->AddCoordinateSystem(coords);
  eavlCellSet *cells = new eavlCellSetAllStructured("cells", reg);
  dataset->AddCellSet(cells);
  eavlFloatArray *field1 = new eavlFloatArray("requestSize", 1, nels);
  eavlFloatArray *field2 = new eavlFloatArray("queueSize", 1, nels);
  dataset->AddField(new eavlField(0, field1, eavlField::ASSOC_CELL_SET,
                 "cells"));
  dataset->AddField(new eavlField(0, field2, eavlField::ASSOC_CELL_SET,
                 "cells"));
                 
  //
  // now loop over the bins and export the files
  //

  int numBins = data[0]->snapshots.size();
  #pragma omp for
  for(int t = 0; t < numBins; t++)
  {
    for(int i = 0; i < numNodes; i++)
    {
      //cout<<"Rsize at i "<<i<<" t "<<t<<" "<<data[i]->snapshots.at(t).requestSize<<endl;
      field1->SetValue(data[i]->cellIndex, data[i]->snapshots.at(t).requestSize);
      field2->SetValue(data[i]->cellIndex, data[i]->snapshots.at(t).queueSize);
    }
    char fileName[50];
    sprintf(fileName,"kripke%04d.vtk", t);
    ofstream outputFile;
    outputFile.open(fileName);
    eavlVTKExporter theExporter(dataset);
    theExporter.Export(outputFile);
  }
  
  
}

void binner(int numBins, NodeData **data, int numNodes)
{
    double totTime = maxTime - minTime;
 
    
    double delta = totTime / float(numBins);
    cout<<"Binning Delta "<<delta<<endl;
    //
    // Current node is the data from an MPI task
    //
    #pragma omp for
    for(int i = 0; i < numNodes; i++)
    {
        //cout<<"current node "<<i<<endl;
        int numberOfSnapshots = data[i]->snapshots.size();
        if(numberOfSnapshots < 1 )
        {
          cerr<<"Node "<<i<<" has no data"<<endl;
        }
        double beginRange = 0;
        double endRange = delta;
        Snapshot *bins = new Snapshot[numBins];
        int binCount[numBins];
        //
        // Init bins
        //
        for(int j = 0; j < numBins; j++)
        {
          bins[j].requestSize = 0;
          bins[j].queueSize = 0;
          bins[j].time = delta * j;
          binCount[j] = 0;
        }
        //
        //  perforn the binning, assumig data is ordered
        //
        int currentBin = 0;
        for(int j = 0; j < numberOfSnapshots; j++)
        {
          Snapshot snapshot = data[i]->snapshots.at(j);
          
          double time = snapshot.time - minTime;
          for(int b = currentBin; b < numBins; b++)
          {
            if((time >= bins[b].time && 
               time < (bins[b].time + delta)) || (b == numBins -1))
            {
              bins[b].requestSize += snapshot.requestSize;
              bins[b].queueSize += snapshot.queueSize;
              binCount[b]++;
              currentBin = b;
              break;
            }
            if(b == (numBins -1)) cerr<<"Failed to find bin "<<time<<" "<<bins[b].time<<endl;
          }
        }
        //
        // average the bins
        //
        for(int j = 0; j < numBins; j++)
        {
          if(binCount[j] != 0)
          {
            bins[j].requestSize = bins[j].requestSize / double(binCount[j]);
            bins[j].queueSize = bins[j].queueSize / double(binCount[j]);
          }
        }
         //clear the current data and put the bin in its place
        data[i]->snapshots.clear();
        if(i == 156)
        {
          cerr<<"156 "<<data[i]->extents[0][0]<<" "<<data[i]->extents[1][0]<<endl;
        }
        for(int b = 0; b < numBins; b++)
        {
          data[i]->snapshots.push_back(bins[b]); 

        }
        delete[] bins;
    }
}

void usage()
{
  cerr<<"Data set creator usage : "<<endl;
  cerr<<"    -f pathToFiles"<<endl;
  cerr<<"    -n numberOfFiles"<<endl; 
  cerr<<"    -b numberOfBins"<<endl; 
}


int main(int argc, char *argv[])
{

  string fileprefix = "/pdata_rank_";
	int numFiles = 32;
	int numBins = 100;
  
  for(int i = 1; i < argc; i++)
  {
    if(strcmp(argv[i],"-f") == 0)
    {
      if(argc <= i + 1)
      {
        cerr<<"No filename specified"<<endl;
        usage();
        return 1;
      }
      fileprefix = argv[++i] + fileprefix;  
      cout<<"Prefix "<<fileprefix<<endl;
    }
    
    if(strcmp(argv[i],"-n") == 0)
    {
      if(argc <= i + 1)
      {
        cerr<<"Missing argument. Need the number of files to process"<<endl;
        usage();
        return 1;
      }
      numFiles = atoi(argv[++i]);
    }
    
    if(strcmp(argv[i],"-b") == 0)
    {
      if(argc <= i + 1)
      {
        cerr<<"Missing argument. Need the number of bins"<<endl;
        usage();
        return 1;
      }
      numBins = atoi(argv[++i]);
    }
    
  }
  NodeData **kripkeData;
  // 
  // Init maxs and mins
  //
  const double MAX = numeric_limits<double>::max();
  const double MIN = numeric_limits<double>::min();
  minTime = minExtents[0] = minExtents[1] = minExtents[2] = MAX;
  maxTime = maxExtents[0] = maxExtents[1] = maxExtents[2] = MIN;
  
  // create the file names to read in

	vector<string> filesToProcess; 
	for(int i = 0; i < numFiles; i++)
	{
	  ostringstream stringStream;
	  stringStream<<fileprefix<<setfill('0')<<setw(4)<<i<<".dat";
	  filesToProcess.push_back(stringStream.str());
	}
	
	kripkeData = new NodeData*[numFiles];
	#pragma omp for
	for(int i = 0; i < numFiles; i++)
	{
	  kripkeData[i] = readFile(filesToProcess.at(i));
	}
	aveTimeDiff /= float(totalTimes);

	cout<<"============== Data Summary ================"<<endl;
	cout<<"Total Nodes  : "<<numFiles<<endl;
	cout<<"Data Extents : <"<<minExtents[0]<<","<<minExtents[1]<<","<<minExtents[2]<<"><"
	                        <<maxExtents[0]<<","<<maxExtents[1]<<","<<maxExtents[2]<<">"<<endl;
	cout<<"minTime      : "<<setprecision(15)<<minTime<<endl;
	cout<<"maxTime      : "<<setprecision(15)<<maxTime<<endl;
	cout<<"Total time   : "<<setprecision(15)<<maxTime - minTime<<" seconds"<<endl; 
	cout<<"Average diff : "<<setprecision(15)<<aveTimeDiff<<endl;
	cout<<"============================================"<<endl;  
	
	binner(numBins, kripkeData, numFiles);
	createDataSets(kripkeData, numFiles);                   
	//
	//  Clean up
	//
	for(int i = 0; i < numFiles; i++)
	{
	  if(kripkeData[i] != NULL) delete kripkeData[i];
	}
	delete[] kripkeData;
	
}


