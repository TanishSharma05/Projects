`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 02/25/2025 11:19:28 AM
// Design Name: 
// Module Name: test_bench_5_1
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module test_bench_5_1();


    timeunit 10ns;
    timeprecision 1ns;
    
    logic		clk;
	logic 		reset;

	logic 		run_i;
	logic 		continue_i;
	logic [15:0] sw_i;

	logic [15:0] led_o;
	logic [7:0]  hex_seg_left;
	logic [3:0]  hex_grid_left;
	logic [7:0]  hex_seg_right;
	logic [3:0]  hex_grid_right;
	
	logic [15:0] expected_ans;
	
	processor_top processor(.*);
	
	initial begin : CLOCK_INITIALIZATION
	       
	   clk = 1;
	end
	
	always begin : CLOCK_GENERATION
	
	   #1 clk = ~clk;
	end
	
	
    //end
//endtask
	
	initial begin : TEST_VECTORS
	

	   run_i <= 0;
	   continue_i <= 0;
	   sw_i <= 16'h0000;
	   reset = 1;
	   repeat(10) @(posedge clk);   
	   reset <= 0;
	   
	   
	   
	   // Test 1: Basic I/O Test 1
//         sw_i = 16'h0003;
         
//         run_i <= 1;
//         #10 run_i <= 0;
//         #50
//         #40 sw_i = 16'h0006;
//         #200;
         
        
        // Test 2: Basic I/O Test 2
//         sw_i = 16'h0006;
//         run_i <= 1;
//         #10 run_i <= 0;
//         #50
//         #20 continue_i = 1;
//         #20 continue_i = 0;
//         #20 continue_i = 1;
//         #20 continue_i = 0;
         
//         sw_i = 16'h0003;
//         #20 continue_i = 1;
//         #20 continue_i = 0;
//         #20 continue_i = 1;
//         #20 continue_i = 0;
//         #200;
        
//        // Test 3: Self-Modifying Code Test
//         sw_i = 16'h000B;
//         run_i <= 1;
//         #10 run_i <= 0;
        
        // Simulate loop iterations with pauses
//         #20 continue_i = 1;
//         #20 continue_i = 0;
//         #20 continue_i = 1;
//         #20 continue_i = 0;
//         #20 continue_i = 1;
//         #20 continue_i = 0;
         
//                  #20 continue_i = 1;
//         #20 continue_i = 0;
//         #20 continue_i = 1;
//         #20 continue_i = 0;
//         #20 continue_i = 1;
//         #20 continue_i = 0;
         
//                           #20 continue_i = 1;
//         #20 continue_i = 0;
//         #20 continue_i = 1;
//         #20 continue_i = 0;
//         #20 continue_i = 1;
//         #20 continue_i = 0;
//         #500;
        
        // Test 4: XOR Test
         
//        sw_i = 16'h0014;
//        run_i <= 1;
//        #10 run_i <= 0;
        
//        #40
//        // Provide first input value
//        sw_i = 16'h0001;
//        #20 continue_i = 1;
//        #20 continue_i = 0;
        
        
//        #40
//        // Provide second input value
//        sw_i = 16'h0002;
//        #20 continue_i = 1;
//        #20 continue_i = 0;
        
//        // Wait for output display
//        #20 continue_i = 1;
//        #20 continue_i = 0;
//        #500;
        
        // Test 5: Multiplication Test
//         sw_i = 16'h0031;
//         run_i <= 1;
//         #10 run_i <= 0;
         
//         #50
         
//         sw_i = 16'h0002;
//         #20 continue_i = 1;
//         #20 continue_i = 0;
//         #20 continue_i = 1;
//         #20 continue_i = 0;
//          #20 continue_i = 1;
//         #20 continue_i = 0;
//         #20 continue_i = 1;
//         #20 continue_i = 0;
//         #20 continue_i = 1;
//         #20 continue_i = 0;
//          #20 continue_i = 1;
//         #20 continue_i = 0;
//         #2000;
        
         //Test 6: Auto Counting Test
         sw_i = 16'h009C;
         run_i <= 1;
         #10 run_i <= 0;
         #200000000;
        
        // Test 7: Sort Test
//       /* #20 reset = 1'b1;
//        #20 reset = 1'b0;
        
//        sw_i = 16'h005A;
        
//        #200 
//        #20 run_i = 1'b1;
//        #20 run_i = 1'b0;
        
        
//        #100
        
        
        
//        #400
//        sw_i = 16'h0002;
        
//        #500
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #100000
//        sw_i = 16'h0003;
        
        
//        #20000
        
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #20 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//        #20 continue_i = 1'b1;
//        #10 continue_i = 1'b0;
        
//         #30000000 $finish();
        

 
        
        

	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   
	   /*sw_i = 16'h0031;
	   
	   run_i <= 1;
	   #10 run_i <= 0;
	   
	   
	   
	   
	   sw_i = 16'h0002;
	   
	   #20 continue_i = 1;
	   #20 continue_i <= 0;
	   
	   #20 continue_i = 1;
	   #20 continue_i <= 0;
	   
	   
	   
	   
	   
	   
	   
	   
	   #1000*/
	    
	   
	   
	   $finish();
	   
	 end
endmodule
