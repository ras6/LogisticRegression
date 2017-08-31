//Simple Logistic regression with some input!

public class LogReg 
{
    int n,nin,nout;
    double[][] W;
    double[] b;
    
    public LogReg(int n, int nin1, int nout1) 
    {
        this.n = n;
        this.nin = nin1;
	this.nout = nout1;
		
        W = new double[this.nout][this.nin];
	b = new double[this.nout];
    }
	
public void toTrain(int[] x, int[] y, double learnr) 
{
    double[] py_x = new double[nout];
    double[] dy = new double[nout];
	
    for(int i=0; i<nout; i++) 
    {
	py_x[i] = 0;
	for(int j=0; j<nin; j++) 
        {
         py_x[i] += W[i][j] * x[j];
        }
        py_x[i] += b[i];
    }
    toSoftmax(py_x);
		
    for(int i=0; i<nout; i++) 
    {
        dy[i] = y[i] - py_x[i];
			
        for(int j=0; j<nin; j++) 
        {
            W[i][j] += learnr * dy[i] * x[j] / n;
        }
			
        b[i] += learnr * dy[i] / n;
	}
}
	
  public void toSoftmax(double[] x) 
  {
   double max1 = 0.0;
   double sum1 = 0.0;
   
   for(int i=0; i<nout; i++)
        if(max1 < x[i]) 
            max1 = x[i];
		
   for(int i=0; i<nout; i++)
    {
        x[i] = Math.exp(x[i] - max1);
        sum1 += x[i];
    }
		
   for(int i=0; i<nout; i++)
      x[i] /= sum1;
  }
	
  public void toPredict(int[] x, double[] y)
  {
         for(int i=0; i<nout; i++) 
           {
		y[i] = 0;
		for(int j=0; j<nin; j++) 
                {
       	          y[i] += W[i][j] * x[j];
		}
		y[i] += b[i];
	    }
		
		toSoftmax(y);
	}	
	
  private static void testLearnRate()
  {
		//Initialization
      double l_rate = 0.1;
      double no_epochs = 500;
		
      int trainN = 6;
      int testN = 2;
      int nin1 = 6;
      int nout1 = 2;
		
      int[][] trainX = {	{1, 0, 1, 0, 1, 1},	{1, 0, 0, 0, 1, 0},	{0, 0, 1, 0, 0, 0},	{0, 0, 0, 0, 1, 0},	{0, 1, 0, 0, 0, 1}, {1, 0, 0, 1, 0, 1}	};
      int[][] trainY = {	{1, 1},	{1, 1},	{1, 0},	{1, 0},	{0, 1},	{0, 1}}; 
		
    //Begining
    LogReg democlass = new LogReg(trainN, nin1, nout1);
		 
    for(int epoch=0; epoch<no_epochs; epoch++)
    {
        for(int i=0; i<trainN; i++)
        {
            democlass.toTrain(trainX[i], trainY[i], l_rate); 
        }
    }
    System.out.println("Learning Rate:"+l_rate);
		
    int[][] testX = {{1, 1, 1, 0, 0, 1}, {0, 1, 0, 0, 1, 1}};
    double[][] testY = new double[testN][nin1];
		
    for(int i=0; i<testN; i++) 
    {
        democlass.toPredict(testX[i], testY[i]);
        for(int j=0; j<nout1; j++) 
        {
            System.out.print(testY[i][j] + " ");
        }
    }
  }
	
  
public static void main(String[] args) 
  {
      testLearnRate();
  }
}
