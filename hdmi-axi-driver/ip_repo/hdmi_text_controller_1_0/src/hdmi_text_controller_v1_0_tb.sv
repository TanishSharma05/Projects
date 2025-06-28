`define SIM_VIDEO 

module hdmi_text_controller_tb();

	//clock and reset_n signals
	logic aclk =1'b0;
	logic arstn = 1'b0;
	
	//Write Address channel (AW)
	logic [31:0] write_addr =32'd0;	//Master write address
	logic [2:0] write_prot = 3'd0;	//type of write(leave at 0)
	logic write_addr_valid = 1'b0;	//master indicating address is valid
	logic write_addr_ready;		    //slave ready to receive address

	//Write Data Channel (W)
	logic [31:0] write_data = 32'd0;	//Master write data
	logic [3:0] write_strb = 4'd0;	    //Master byte-wise write strobe
	logic write_data_valid = 1'b0;	    //Master indicating write data is valid
	logic write_data_ready;		        //slave ready to receive data

	//Write Response Channel (WR)
	logic write_resp_ready = 1'b0;	//Master ready to receive write response
	logic [1:0] write_resp;		    //slave write response
	logic write_resp_valid;		    //slave response valid
	
	//Read Address channel (AR)
	logic [31:0] read_addr = 32'd0;	//Master read address
	logic [2:0] read_prot =3'd0;	//type of read(leave at 0)
	logic read_addr_valid = 1'b0;	//Master indicating address is valid
	logic read_addr_ready;		    //slave ready to receive address

	//Read Data Channel (R)
	logic read_data_ready = 1'b0;	//Master indicating ready to receive data
	logic [31:0] read_data;		    //slave read data
	logic [1:0] read_resp;		    //slave read response
	logic read_data_valid;		    //slave indicating data in channel is valid

    
    logic [3:0] pixel_rgb [3];
    logic pixel_clk, pixel_hs, pixel_vs, pixel_vde;
    logic [9:0] drawX, drawY;
    logic [31:0] tb_read;
    
    //BMP writer related signals    
    localparam BMP_WIDTH  = 800;
    localparam BMP_HEIGHT = 525;
    localparam int BASE = 0;

    logic [23:0] bitmap [BMP_WIDTH][BMP_HEIGHT];

    integer i,j; 

	hdmi_text_controller_v1_0 # (
		.C_AXI_DATA_WIDTH(32),
		.C_AXI_ADDR_WIDTH(16)
	) hdmi_text_controller_v1_0_inst (

		.axi_aclk(aclk),
		.axi_aresetn(arstn),

		.axi_awaddr(write_addr),
		.axi_awprot(write_prot),
		.axi_awvalid(write_addr_valid),
		.axi_awready(write_addr_ready),

		.axi_wdata(write_data),
		.axi_wstrb(write_strb),
		.axi_wvalid(write_data_valid),
		.axi_wready(write_data_ready),

		.axi_bresp(write_resp),
		.axi_bvalid(write_resp_valid),
		.axi_bready(write_resp_ready),

		.axi_araddr(read_addr),
		.axi_arprot(read_prot),
		.axi_arvalid(read_addr_valid),
		.axi_arready(read_addr_ready),

		.axi_rdata(read_data),
		.axi_rresp(read_resp),
		.axi_rvalid(read_data_valid),
		.axi_rready(read_data_ready)
	);
	
	initial begin: CLOCK_INITIALIZATION
	   aclk = 1'b1;
    end 
       
    always begin : CLOCK_GENERATION
        #5 aclk = ~aclk;
    end
    
     assign pixel_rgb[0] = hdmi_text_controller_v1_0_inst.red;
     assign pixel_rgb[1] = hdmi_text_controller_v1_0_inst.green;
     assign pixel_rgb[2] = hdmi_text_controller_v1_0_inst.blue;
    
     assign pixel_clk = hdmi_text_controller_v1_0_inst.clk_25MHz;
     assign pixel_hs = hdmi_text_controller_v1_0_inst.hsync;
     assign pixel_vs = hdmi_text_controller_v1_0_inst.vsync;
     assign pixel_vde = hdmi_text_controller_v1_0_inst.vde;
    
     assign drawX = hdmi_text_controller_v1_0_inst.drawX;
     assign drawY = hdmi_text_controller_v1_0_inst.drawY;
   
    
    task save_bmp(string bmp_file_name);
        begin
        
            integer unsigned        fout_bmp_pointer, BMP_file_size,BMP_row_size,r;
            logic   unsigned [31:0] BMP_header[0:12];
        
                                      BMP_row_size  = 32'(BMP_WIDTH) & 32'hFFFC;  
        if ((BMP_WIDTH & 32'd3) !=0)  BMP_row_size  = BMP_row_size + 4;           
    
        fout_bmp_pointer= $fopen(bmp_file_name,"wb");
        if (fout_bmp_pointer==0) begin
            $display("Could not open file '%s' for writing",bmp_file_name);
            $stop;     
        end
        $display("Saving bitmap '%s'.",bmp_file_name);
       
        BMP_header[0:12] = '{BMP_file_size,0,0054,40,BMP_WIDTH,BMP_HEIGHT,{16'd24,16'd8},0,(BMP_row_size*BMP_HEIGHT*3),2835,2835,0,0};
        
        //Write header out      
        $fwrite(fout_bmp_pointer,"BM");
        for (int i =0 ; i <13 ; i++ ) $fwrite(fout_bmp_pointer,"%c%c%c%c",BMP_header[i][7 -:8],BMP_header[i][15 -:8],BMP_header[i][23 -:8],BMP_header[i][31 -:8]); // Better compatibility with Lattice Active_HDL.
        
        for (int y=BMP_HEIGHT-1;y>=0;y--) begin
          for (int x=0;x<BMP_WIDTH;x++)
            $fwrite(fout_bmp_pointer,"%c%c%c",bitmap[x][y][23:16],bitmap[x][y][15:8],bitmap[x][y][7:0]) ;
        end
    
        $fclose(fout_bmp_pointer);
        end
    endtask
    
    always@(posedge pixel_clk)
        if (!arstn) begin
            for (j = 0; j < BMP_HEIGHT; j++)    
                for (i = 0; i < BMP_WIDTH; i++) 
                    bitmap[i][j] <= 24'h0F0F0F; 
        end
        else
            if (pixel_vde) 
                bitmap[drawX][drawY] <= {pixel_rgb[0], 4'h0, pixel_rgb[1], 4'h0, pixel_rgb[2], 4'h00};
  
    task axi_write (input logic [31:0] addr, input logic [31:0] data);
        begin
            #3 write_addr <= addr;	
            write_data <= data;	
            write_addr_valid <= 1'b1;	
            write_data_valid <= 1'b1;	
            write_resp_ready <= 1'b1;	
            write_strb <= 4'hF;		
    
            wait(write_data_ready || write_addr_ready);
                
            @(posedge aclk); 
            if(write_data_ready&&write_addr_ready)
            begin
                write_addr_valid<=0;
                write_data_valid<=0;
            end
            else    
            begin
                if(write_data_ready)    //case data handshake completed
                begin
                    write_data_valid<=0;
                    wait(write_addr_ready); //wait for address address ready
                end
                        else if(write_addr_ready)   //case address handshake completed
                        begin
                    write_addr_valid<=0;
                            wait(write_data_ready); //wait for data ready
                        end 
                @ (posedge aclk);// complete the second handshake
                write_addr_valid<=0; //make sure both valid signals are deasserted
                write_data_valid<=0;
            end
                
            //both handshakes have occured
            //deassert strobe
            write_strb<=0;
    
            //wait for valid response
            wait(write_resp_valid);
            
            //both handshake signals and rising edge
            @(posedge aclk);
    
            //deassert ready for response
            write_resp_ready<=0;
    
            //end of write transaction
        end
    endtask;
    
    task axi_read (input logic [31:0] addr, output logic [31:0] data);
        begin
            #3;
            read_addr <= addr;
            read_addr_valid <= 1'b1;
            read_data_ready <= 1'b1;
            wait (read_addr_ready);
            @(posedge aclk);
            read_addr_valid <= 1'b0;
            wait (read_data_valid);
            @(posedge aclk);
            data <= read_data;
            read_data_ready <= 1'b0;
        end
    endtask;
  
  
    initial begin: TEST_VECTORS
        arstn = 0; //reset IP
        repeat (4) @(posedge aclk);
        arstn <= 1;
        
        repeat (4) @(posedge aclk) axi_write((600*4), 32'h001F6004); //write control reg to set foreground and background
        

    for(i = 0; i < 8; i++) begin
        repeat (4) @(posedge aclk) axi_write(32'h2000 + (i*4), 32'h0000_0000);
    end

    // Palette Register 0: Black (0) and Blue (1)
    repeat (4) @(posedge aclk) axi_write(32'h2000, 
        ((4'hF << 21) | (4'h0 << 17) | (4'h0 << 13)) |  // Blue
        ((4'h0 << 9)  | (4'h0 << 5)  | (4'h0 << 1)));   // Black

// Palette Register 1: White (2) and Green (3)
    repeat (4) @(posedge aclk) axi_write(32'h2004,
        ((4'h0 << 21) | (4'hF << 17) | (4'h0 << 13)) |  // Green
        ((4'hF << 9)  | (4'hF << 5)  | (4'hF << 1)));   // White

// Palette Register 2: Gray (4)
    repeat (4) @(posedge aclk) axi_write(32'h2008,
        ((4'h8 << 21) | (4'h8 << 17) | (4'h8 << 13)));  // Gray

    // Clear VRAM
    for(i = 0; i < 600; i++) begin
        repeat (4) @(posedge aclk) axi_write(4*i, 32'h0000_0000);
    end


    repeat (4) @(posedge aclk) axi_write(32'h0000, {1'b1,7'h61,4'h0,4'h1, 1'b0,7'h74,4'h1,4'h0}); // "ta"
    repeat (4) @(posedge aclk) axi_write(32'h0004, {1'b1,7'h69,4'h0,4'h1, 1'b0,7'h6E,4'h1,4'h0}); // "ni"
    repeat (4) @(posedge aclk) axi_write(32'h0008, {1'b1,7'h68,4'h0,4'h1, 1'b0,7'h73,4'h1,4'h0}); // "sh"
    repeat (4) @(posedge aclk) axi_write(32'h000C, {1'b1,7'h32,4'h0,4'h1, 1'b0,7'h73,4'h1,4'h0}); // "s2"
    repeat (4) @(posedge aclk) axi_write(32'h0010, {1'b1,7'h61,4'h0,4'h1, 1'b0,7'h20,4'h1,4'h0}); // " a"
    repeat (4) @(posedge aclk) axi_write(32'h0014, {1'b1,7'h64,4'h0,4'h1, 1'b0,7'h6E,4'h1,4'h0}); // "nd"
    repeat (4) @(posedge aclk) axi_write(32'h0018, {1'b1,7'h67,4'h0,4'h1, 1'b0,7'h20,4'h1,4'h0}); // " g"
    repeat (4) @(posedge aclk) axi_write(32'h001C, {1'b1,7'h72,4'h0,4'h1, 1'b0,7'h61,4'h1,4'h0}); // "ar"
    repeat (4) @(posedge aclk) axi_write(32'h0020, {1'b1,7'h6B,4'h0,4'h1, 1'b0,7'h76,4'h1,4'h0}); // "vk"
    repeat (4) @(posedge aclk) axi_write(32'h0024, {1'b1,7'h20,4'h0,4'h1, 1'b0,7'h32,4'h1,4'h0}); // "2 "
    
    repeat (4) @(posedge aclk) axi_write(32'h0028, {1'b1,7'h6F,4'h0,4'h2, 1'b0,7'h63,4'h2,4'h0}); // "Co"
    repeat (4) @(posedge aclk) axi_write(32'h002C, {1'b1,7'h70,4'h0,4'h2, 1'b0,7'h6D,4'h2,4'h0}); // "mp"
    repeat (4) @(posedge aclk) axi_write(32'h0030, {1'b1,7'h65,4'h0,4'h2, 1'b0,7'h6C,4'h2,4'h0}); // "le"
    repeat (4) @(posedge aclk) axi_write(32'h0034, {1'b1,7'h65,4'h0,4'h2, 1'b0,7'h74,4'h2,4'h0}); // "te"
    repeat (4) @(posedge aclk) axi_write(32'h0038, {1'b1,7'h20,4'h0,4'h2, 1'b0,7'h64,4'h2,4'h0}); // "d "
    
    repeat (4) @(posedge aclk) axi_write(32'h003C, {1'b1,7'h43,4'h0,4'h3, 1'b0,7'h45,4'h3,4'h0}); // "EC"
    repeat (4) @(posedge aclk) axi_write(32'h0040, {1'b1,7'h33,4'h0,4'h3, 1'b0,7'h45,4'h3,4'h0}); // "E3"
    repeat (4) @(posedge aclk) axi_write(32'h0044, {1'b1,7'h35,4'h0,4'h3, 1'b0,7'h38,4'h3,4'h0}); // "85"
    repeat (4) @(posedge aclk) axi_write(32'h0048, {1'b1,7'h20,4'h0,4'h3, 1'b0,7'h21,4'h3,4'h0}); // "! "

 
		`ifdef SIM_VIDEO
		wait (~pixel_vs);
		save_bmp ("lab7_2_sim.bmp");
		`endif
		$finish();
	end
    
endmodule	

	