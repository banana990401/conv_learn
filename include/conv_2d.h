typedef struct
{
    float*   input;                                   
    float*   weight;                                  
    float*   bias;                                    
    float*   output;                                  
    unsigned int      n;                                  
    unsigned int      c;                                  
    unsigned int      h;                                  
    unsigned int      w;                                  
    unsigned int      k;                                  
    unsigned int      r;                                  
    unsigned int      s;                                  
    unsigned int      u;                                  
    unsigned int      v;                                  
    unsigned int      p;                                  
    unsigned int      q;                                  
    unsigned int      Oh;                                
    unsigned int      Ow;                             
}param_t;
void launch_implgemm(param_t param);
