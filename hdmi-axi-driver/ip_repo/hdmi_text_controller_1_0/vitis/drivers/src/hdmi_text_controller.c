

#include "hdmi_text_controller.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "sleep.h"


void paletteTest()
{
	textHDMIColorClr();

	for (int i = 0; i < 8; i ++)
	{
		char color_string[80];
		sprintf(color_string, "Foreground: %d background %d", 2*i, 2*i+1);
		textHDMIDrawColorText (color_string, 0, 2*i, 2*i, 2*i+1);
		sprintf(color_string, "Foreground: %d background %d", 2*i+1, 2*i);
		textHDMIDrawColorText (color_string, 40, 2*i, 2*i+1, 2*i);
	}
	textHDMIDrawColorText ("The above text should cycle through random colors", 0, 25, 0, 1);


	for (int i = 0; i < 10; i++)
	{
		sleep_MB (1);
		for (int j = 0; j < 16; j++)
			setColorPalette(j, 	rand() % 16, rand() % 16,rand() % 16); 

	}
}

void textHDMIColorClr()
{
	for (int i = 0; i<(ROWS*COLUMNS) * 2; i++)
	{
		hdmi_ctrl->VRAM[i] = 0x00;
	}
}

void textHDMIDrawColorText(char* str, int x, int y, uint8_t background, uint8_t foreground)
{
	int i = 0;
	while (str[i]!=0)
	{
		hdmi_ctrl->VRAM[(y*COLUMNS + x + i) * 2] = foreground << 4 | background;
		hdmi_ctrl->VRAM[(y*COLUMNS + x + i) * 2 + 1] = str[i];
		i++;
	}
}

void setColorPalette (uint8_t color, uint8_t red, uint8_t green, uint8_t blue)
{
	volatile uint32_t* pal_ptr = (volatile uint32_t*)((uint8_t*)hdmi_ctrl + 0x2000);
	uint32_t index = color >> 1;
	uint32_t prev_val = pal_ptr[index];
	uint32_t updated_val = (color & 1)
	    ? (prev_val & 0x00001FFF) | ((red & 0xF) << 21) | ((green & 0xF) << 17) | ((blue & 0xF) << 13)
	    : (prev_val & 0x1FFF0000) | ((red & 0xF) << 9) | ((green & 0xF) << 5) | ((blue & 0xF) << 1);
	pal_ptr[index] = updated_val;
}

void textHDMIColorScreenSaver()
{
	paletteTest();
	char color_string[80];
    int fg, bg, x, y;
	textHDMIColorClr();
	//initialize palette
	for (int i = 0; i < 16; i++)
	{
		setColorPalette (i, colors[i].red, colors[i].green, colors[i].blue);
	}
	while (1)
	{
		fg = rand() % 16;
		bg = rand() % 16;
		while (fg == bg)
		{
			fg = rand() % 16;
			bg = rand() % 16;
		}
		sprintf(color_string, "Drawing %s text with %s background", colors[fg].name, colors[bg].name);
		x = rand() % (80-strlen(color_string));
		y = rand() % 30;
		textHDMIDrawColorText (color_string, x, y, bg, fg);
		sleep_MB (1);
	}
}

hdmiTest()
{
    //On-chip memory write and readback test
	uint32_t checksum[ROWS], readsum[ROWS];

	for (int j = 0; j < ROWS; j++)
	{
		checksum[j] = 0;
		for (int i = 0; i < COLUMNS * 2; i++)
		{
			hdmi_ctrl->VRAM[j*COLUMNS*2 + i] = i + j;
			checksum[j] += i + j;
		}
	}
	
	for (int j = 0; j < ROWS; j++)
	{
		readsum[j] = 0;
		for (int i = 0; i < COLUMNS * 2; i++)
		{
			readsum[j] += hdmi_ctrl->VRAM[j*COLUMNS*2 + i];
			//printf ("%x \n\r", hdmi_ctrl->VRAM[j*COLUMNS*2 + i]);
		}
		printf ("Row: %d, Checksum: %x, Read-back Checksum: %x\n\r", j, checksum[j], readsum[j]);
		if (checksum[j] != readsum[j])
		{
			printf ("Checksum mismatch!, check your AXI4 code or your on-chip memory\n\r");
			while (1){};
		}
	}
	printf ("Checksum passed, beginning palette test\n\r");
	
	paletteTest();
	printf ("Palette test passed, beginning screensaver loop\n\r");
    textHDMIColorScreenSaver();
}

