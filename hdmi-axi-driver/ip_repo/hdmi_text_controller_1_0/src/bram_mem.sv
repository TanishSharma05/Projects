`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/07/2025 05:08:39 PM
// Design Name: 
// Module Name: bram_mem
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


///ajkahjljkjkjdaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa


module bram_mem(
    
    input logic clk,
    input logic reset,
    input logic [31:0] data,
    
    input logic [10:0] waddr,
    input logic [10:0] raddr,
    input logic [3:0] wen,
    input logic [10:0] final_read_addr,
    output logic [31:0] rout,
    output logic [31:0] final_rout

    );
    
    logic [10:0] addr_;
    logic [31:0] data_;
    logic [3:0] wren_;
    logic [10:0] addra;
    logic [10:0] addrb;
    logic [31:0] dina;
    logic [31:0] dinb;
    logic [31:0] douta;
    logic [31:0] doutb;
    logic ena;
    logic enb;
    logic [3:0] wea;
    logic [3:0] web;
    
    instantiate_ram init ( 
	   .reset(reset),
	   .clk(clk),

	   .addr(addr_),
	   .wren(wren_),
	   .data(data_)
    );
    
    always_comb begin
    
        if(wren_) begin
            
            wea <= wren_;
            ena <= 1;
            dina <= data_;
            addra <= addr_;
        end
        
        else if(wen) begin
            wea <= wen;
            ena = 1;
            dina <= data;
            addra <= waddr;
        end
        
        else begin
            wea <= 0;
            ena <= 1;
            dina <= 0;
            addra <= final_read_addr;
        end
        
        
    end
    


    blk_mem_gen_0 bram_inst (
    
        .addra(addra), 
        .addrb(raddr),
        .dina(dina),
//        .dinb(dinb),
        .douta(rout),
        .doutb(final_rout),
        .ena(ena),
        .enb(1'b1),
        .wea(wea),
        .web(4'b0000),
        .clka(clk),
        .clkb(clk)
    );
endmodule
