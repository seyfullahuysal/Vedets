#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.cpp"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct svm_model* training();
double test(float [],svm_model *);

struct svm_parameter param;     // set by parse_command_line
struct svm_problem prob;        // set by read_problem
struct svm_model *model;
struct svm_node *x_space;

int main(int argc, char **argv)
{
    struct svm_model *model2=training();
	
	float test_data[]={0.00588,0.00027,0.000193,1.148294,0.000211,0.000288,6.743363};
	printf("Aracin Sinifi=%f",test(test_data,model2));
    return 0;
}


svm_model* training()
{
	
	char input_file_name[1024];
    char model_file_name[1024];
    const char *error_msg;
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = 0.5;
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;


    //Problem definition-------------------------------------------------------------
    prob.l = 6753;
    int ozellik_sayisi=7;
    //x values matrix of xor values
    
    FILE *dosya;
	if((dosya=fopen("vehicle2.arff","r"))==NULL)
	{
		
		printf("dosya acilamadi 5 !!!");
		exit(0);
		
	}
    
    float matrix[prob.l][ozellik_sayisi];
	float y[prob.l];
	int i=0;	
	while(!feof(dosya))
	{
		fscanf(dosya,"%f %f %f %f %f %f %f %f\n",&matrix[i][0],&matrix[i][1],&matrix[i][2],&matrix[i][3],&matrix[i][4],&matrix[i][5],&matrix[i][6],&y[i]);
		i++;
	}
    
    //This part i have trouble understanding
    svm_node** x = Malloc(svm_node*,prob.l);

    //Trying to assign from matrix to svm_node training examples
    for (int row = 0;row <prob.l; row++){
        svm_node* x_space = Malloc(svm_node,ozellik_sayisi+1);
        for (int col = 0;col < ozellik_sayisi;col++){
            x_space[col].index = col;
            x_space[col].value = matrix[row][col];
        }
        x_space[ozellik_sayisi].index = -1;      //Each row of properties should be terminated with a -1 according to the readme
        x[row] = x_space;
    }

    prob.x = x;

    //yvalues
    prob.y = Malloc(double,prob.l);
    
    
    for(int i=0;i<prob.l;i++)
    {
    	prob.y[i]=y[i];
	}

    //Train model---------------------------------------------------------------------
    model = svm_train(&prob,&param);
    svm_destroy_param(&param);
    free(prob.y);
    free(prob.x);
    free(x_space);
    return model;
	
}




double test(float ozellik[],struct svm_model *model)
{
	 //Test model----------------------------------------------------------------------
    svm_node* testnode = Malloc(svm_node,7);
    testnode[0].index = 0;
    testnode[0].value = ozellik[0];
    testnode[1].index = 1;
	testnode[1].value = ozellik[1];	
	testnode[2].index = 2;
    testnode[2].value = ozellik[2];
    testnode[3].index = 3;
    testnode[3].value = ozellik[3];
    testnode[4].index = 4;
    testnode[4].value = ozellik[4];
    testnode[5].index = 5;
    testnode[5].value = ozellik[5];
    testnode[6].index = 6;
    testnode[6].value = ozellik[6];
    testnode[7].index = -1;
    //This works correctly:
    double retval = svm_predict(model,testnode);
    return retval;

}
