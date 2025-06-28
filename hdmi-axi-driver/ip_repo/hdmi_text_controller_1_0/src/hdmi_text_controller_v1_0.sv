`timescale 1 ns / 1 ps

module hdmi_text_controller_v1_0 #
(
    // Parameters of Axi Slave Bus Interface S00_AXI
    // Modify parameters as necessary for access of full VRAM range

    parameter integer C_AXI_DATA_WIDTH	= 32,
    parameter integer C_AXI_ADDR_WIDTH	= 14
)
(
    // Users to add ports here

    output logic hdmi_clk_n,
    output logic hdmi_clk_p,
    output logic [2:0] hdmi_tx_n,
    output logic [2:0] hdmi_tx_p,

    // User ports ends
    // Do not modify the ports beyond this line


    // Ports of Axi Slave Bus Interface AXI
    input logic  axi_aclk,
    input logic  axi_aresetn,
    input logic [C_AXI_ADDR_WIDTH-1 : 0] axi_awaddr,
    input logic [2 : 0] axi_awprot,
    input logic  axi_awvalid,
    output logic  axi_awready,
    input logic [C_AXI_DATA_WIDTH-1 : 0] axi_wdata,
    input logic [(C_AXI_DATA_WIDTH/8)-1 : 0] axi_wstrb,
    input logic  axi_wvalid,
    output logic  axi_wready,
    output logic [1 : 0] axi_bresp,
    output logic  axi_bvalid,
    input logic  axi_bready,
    input logic [C_AXI_ADDR_WIDTH-1 : 0] axi_araddr,
    input logic [2 : 0] axi_arprot,
    input logic  axi_arvalid,
    output logic  axi_arready,
    output logic [C_AXI_DATA_WIDTH-1 : 0] axi_rdata,
    output logic [1 : 0] axi_rresp,
    output logic  axi_rvalid,
    input logic  axi_rready
);

//additional logic variables as necessary to support VGA, and HDMI modules.
logic        clk_25MHz;
logic        clk_125MHz;
logic        locked;
logic        reset_ah;
logic        tyemp;

assign reset_ah = ~axi_aresetn;
logic [3:0]  red, green, blue;
logic        hsync, vsync, vde;
logic [9:0]  drawX, drawY;
//logic [9:0]  BallX, BallY;
//logic [3:0]  Ball_size; 

logic [7:0]  font_data;
logic [10:0] font_rom_address;
logic        is_inverted;
logic [31:0] ctrl_reg;

logic        hdmi_tmds_clk_p, hdmi_tmds_clk_n;
logic [2:0]  hdmi_tmds_data_p, hdmi_tmds_data_n;

logic [31:0] rout;
logic [10:0] raddr;
logic [15:0] char;
logic [31:0] palette [8];
logic [10:0] write_addr;
    
logic [31:0] write_data;
logic [3:0] write_ena;
    
logic [10:0] read_addr;
    
logic [31:0] read_data;


always_comb begin

    raddr = (((drawY >> 4) * 80) + (drawX >> 3)) >> 1;
    
    if(drawX[3] == 0)
        char = rout[15:0];
    else
        char = rout[31:16];  
end


bram_mem bram(
    
    .clk(axi_aclk),
    .reset(~axi_aresetn),
    .data(write_data),
    
    .waddr(write_addr),
    .raddr(raddr),
    .wen(write_ena),
    .final_read_addr(read_addr),
    .rout(read_data),
    .final_rout(rout)
);

// Instantiation of Axi Bus Interface AXI
hdmi_text_controller_v1_0_AXI # ( 
    .C_S_AXI_DATA_WIDTH(C_AXI_DATA_WIDTH),
    .C_S_AXI_ADDR_WIDTH(C_AXI_ADDR_WIDTH)
) hdmi_text_controller_v1_0_AXI_inst (
    .S_AXI_ACLK(axi_aclk),
    .S_AXI_ARESETN(axi_aresetn),
    .S_AXI_AWADDR(axi_awaddr),
    .S_AXI_AWPROT(axi_awprot),
    .S_AXI_AWVALID(axi_awvalid),
    .S_AXI_AWREADY(axi_awready),
    .S_AXI_WDATA(axi_wdata),
    .S_AXI_WSTRB(axi_wstrb),
    .S_AXI_WVALID(axi_wvalid),
    .S_AXI_WREADY(axi_wready),
    .S_AXI_BRESP(axi_bresp),
    .S_AXI_BVALID(axi_bvalid),
    .S_AXI_BREADY(axi_bready),
    .S_AXI_ARADDR(axi_araddr),
    .S_AXI_ARPROT(axi_arprot),
    .S_AXI_ARVALID(axi_arvalid),
    .S_AXI_ARREADY(axi_arready),
    .S_AXI_RDATA(axi_rdata),
    .S_AXI_RRESP(axi_rresp),
    .S_AXI_RVALID(axi_rvalid),
    .S_AXI_RREADY(axi_rready),
    .DrawX(drawX),
    .DrawY(drawY),
    .is_inverted(is_inverted),
    .font_rom_address(font_rom_address),
    .ctrl_reg(ctrl_reg),
    
     .palette(palette),
    
    .write_addr(write_addr),
    
    .write_data(write_data),
    
    .write_ena(write_ena),
    
    .read_addr(read_addr),
    
    .read_data(read_data)
);

//font_rom #(

//    .ADDR_WIDTH(11),
//    .DATA_WIDTH(8)
//) font_rom_inst (

//    .addr(font_rom_address),
//    .data(font_data)
//);



clk_wiz_0 clk_wiz (
        .clk_out1(clk_25MHz),
        .clk_out2(clk_125MHz),
        .reset(reset_ah),
        .locked(locked),
        .clk_in1(axi_aclk)
);

//Real Digital VGA to HDMI converter
hdmi_tx_1 vga_to_hdmi (
        //Clocking and Reset
        .pix_clk(clk_25MHz),
        .pix_clkx5(clk_125MHz),
        .pix_clk_locked(locked),
        //Reset is active LOW
        .rst(reset_ah),
        //Color and Sync Signals
        .red(red),
        .green(green),
        .blue(blue),
        .hsync(hsync),
        .vsync(vsync),
        .vde(vde),
        
        //aux Data (unused)
        .aux0_din(4'b0),
        .aux1_din(4'b0),
        .aux2_din(4'b0),
        .ade(1'b0),
        
        //Differential outputs
        .TMDS_CLK_P(hdmi_clk_p),          
        .TMDS_CLK_N(hdmi_clk_n),          
        .TMDS_DATA_P(hdmi_tx_p),         
        .TMDS_DATA_N(hdmi_tx_n)          
    );


 //VGA Sync signal generator
    vga_controller vga (
        .pixel_clk(clk_25MHz),
        .reset(reset_ah),
        .hs(hsync),
        .vs(vsync),
        .active_nblank(vde),
        .drawX(drawX),
        .drawY(drawY)
    );    

//Color Mapper Module   
    color_mapper color_instance(
        .BallX(0),
        .BallY(0),
        .char(char),
        .palette(palette),
        .DrawX(drawX),
        .DrawY(drawY),
        .Ball_size(0),
        .Red(red),
        .Green(green),
        .Blue(blue),
        .is_inverted(char[15]),
        .ctrl_reg(ctrl_reg)
//        .font_data(font_data)
    );



// User logic ends

endmodule
