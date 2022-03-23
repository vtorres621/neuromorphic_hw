`timescale 1ns / 1ps

module mnist_cnn(image, conv1_weight, conv2_weight, fc1_weight, clk, class);
    input [6272:0]image;
    input [6400:0] conv1_weight; //32 x 1 x 5 x 5
    input [51100:0] conv2_weight [7:0]; //64 x 32 x 5 x 5 
    input [10239:0] fc1_weight [7:0]; //10 x 1024
    input clk;
    output reg [3:0]class;
    
    reg []accum0;
    
    
    
    always @(posedge clk)
    begin
                    
    end 
    
    
endmodule
