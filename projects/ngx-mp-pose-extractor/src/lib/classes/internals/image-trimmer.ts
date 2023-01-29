import { lstat } from 'fs';

export class ImageTrimmer {
  private canvas?: HTMLCanvasElement;
  private context?: CanvasRenderingContext2D;

  constructor() {}

  async loadByDataUrl(dataUrl: string) {
    const image: HTMLImageElement = await new Promise((resolve, reject) => {
      const image = new Image();
      image.src = dataUrl;
      image.onload = () => {
        resolve(image);
      };
    });

    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d')!;
    canvas.width = image.width;
    canvas.height = image.height;
    context.drawImage(image, 0, 0);

    this.canvas = canvas;
    this.context = context;
  }

  async trimMargin(marginColor: string, diffThreshold: number = 10) {
    if (this.canvas === undefined) throw new Error('Image is not loaded');

    // マージンを検出する範囲を指定 (左端から0〜20%)
    const edgeDetectionRangeMinX = 0;
    const edgeDetectionRangeMaxX = Math.floor(this.canvas.width * 0.2);

    // マージンの端を検出
    const edgePositionFromTop = await this.getVerticalEdgePositionOfColor(
      marginColor,
      'top',
      diffThreshold,
      edgeDetectionRangeMinX,
      edgeDetectionRangeMaxX
    );
    const marginTop = edgePositionFromTop != null ? edgePositionFromTop : 0;

    const edgePositionFromBottom = await this.getVerticalEdgePositionOfColor(
      marginColor,
      'bottom',
      diffThreshold,
      edgeDetectionRangeMinX,
      edgeDetectionRangeMaxX
    );
    const marginBottom =
      edgePositionFromBottom != null
        ? edgePositionFromBottom
        : this.canvas.height;

    const oldHeight = this.canvas.height;
    const newHeight = marginBottom - marginTop;
    this.crop(0, marginTop, this.canvas.width, newHeight);

    return {
      marginTop: marginTop,
      marginBottom: marginBottom,
      heightNew: newHeight,
      heightOld: oldHeight,
      width: this.canvas.width,
    };
  }

  async crop(x: number, y: number, w: number, h: number) {
    if (!this.canvas || !this.context) return;

    const newCanvas = document.createElement('canvas');
    const newContext = newCanvas.getContext('2d')!;
    newCanvas.width = w;
    newCanvas.height = h;
    newContext.drawImage(
      this.canvas,
      x,
      y,
      newCanvas.width,
      newCanvas.height,
      0,
      0,
      newCanvas.width,
      newCanvas.height
    );
    this.replaceCanvas(newCanvas);
  }

  async replaceColor(
    srcColor: string,
    dstColor: string,
    diffThreshold: number
  ) {
    if (!this.canvas || !this.context) return;

    const imageData = this.context.getImageData(
      0,
      0,
      this.canvas.width,
      this.canvas.height
    );

    const dstColorValue = this.hexColorCodeToRgba(dstColor);

    for (let x = 0; x < imageData.width; x++) {
      for (let y = 0; y < imageData.height; y++) {
        const idx = (x + y * imageData.width) * 4;
        const red = imageData.data[idx + 0];
        const green = imageData.data[idx + 1];
        const blue = imageData.data[idx + 2];
        const alpha = imageData.data[idx + 3];

        const colorCode = this.rgbToHexColorCode(red, green, blue);
        if (!this.isSimilarColor(srcColor, colorCode, diffThreshold)) {
          continue;
        }

        imageData.data[idx + 0] = dstColorValue.r;
        imageData.data[idx + 1] = dstColorValue.g;
        imageData.data[idx + 2] = dstColorValue.b;
        if (dstColorValue.a !== undefined) {
          imageData.data[idx + 3] = dstColorValue.a;
        }
      }
    }

    this.context.putImageData(imageData, 0, 0);
  }

  async getMarginColor() {
    if (!this.canvas || !this.context) {
      return null;
    }
    let marginColor: string | null = null;

    const imageData = this.context.getImageData(
      0,
      0,
      this.canvas.width,
      this.canvas.height
    );

    let isBreak = false;
    for (let x = 0; x < imageData.width && !isBreak; x++) {
      for (let y = 0; y < imageData.height; y++) {
        const idx = (x + y * imageData.width) * 4;
        const red = imageData.data[idx + 0];
        const green = imageData.data[idx + 1];
        const blue = imageData.data[idx + 2];
        const alpha = imageData.data[idx + 3];

        const colorCode = this.rgbToHexColorCode(red, green, blue);
        if (marginColor != colorCode) {
          if (marginColor === null) {
            marginColor = colorCode;
          } else {
            isBreak = true;
            break;
          }
        }
      }
    }

    return marginColor;
  }

  async getVerticalEdgePositionOfColor(
    color: string,
    direction: 'top' | 'bottom',
    diffThreshold: number,
    minX?: number,
    maxX?: number
  ) {
    if (!this.canvas || !this.context) {
      return null;
    }

    if (minX === undefined) {
      minX = 0;
    }
    if (maxX === undefined) {
      maxX = this.canvas.width;
    }

    const imageData = this.context.getImageData(
      0,
      0,
      this.canvas.width,
      this.canvas.height
    );

    let isBreak = false;

    let edgePositionY;
    if (direction === 'top') {
      edgePositionY = 0;

      for (let y = 0; y < imageData.height; y++) {
        for (let x = 0; x < maxX && !isBreak; x++) {
          const idx = (x + y * imageData.width) * 4;
          const red = imageData.data[idx + 0];
          const green = imageData.data[idx + 1];
          const blue = imageData.data[idx + 2];
          const alpha = imageData.data[idx + 3];

          const colorCode = this.rgbToHexColorCode(red, green, blue);
          if (
            color == colorCode ||
            this.isSimilarColor(color, colorCode, diffThreshold)
          ) {
            if (edgePositionY < y) {
              edgePositionY = y;
            }
          } else {
            isBreak = true;
            break;
          }
        }
      }

      if (edgePositionY !== 0) {
        edgePositionY += 1;
      }
    } else if (direction === 'bottom') {
      edgePositionY = this.canvas.height;

      for (let y = imageData.height - 1; y >= 0; y--) {
        for (let x = 0; x < imageData.width && !isBreak; x++) {
          const idx = (x + y * imageData.width) * 4;
          const red = imageData.data[idx + 0];
          const green = imageData.data[idx + 1];
          const blue = imageData.data[idx + 2];
          const alpha = imageData.data[idx + 3];

          const colorCode = this.rgbToHexColorCode(red, green, blue);
          if (
            color == colorCode ||
            this.isSimilarColor(color, colorCode, diffThreshold)
          ) {
            if (edgePositionY > y) {
              edgePositionY = y;
            }
          } else {
            isBreak = true;
            break;
          }
        }
      }

      if (edgePositionY !== this.canvas.height) {
        edgePositionY -= 1;
      }
    }

    return edgePositionY;
  }

  async getWidth() {
    return this.canvas?.width;
  }

  async getHeight() {
    return this.canvas?.height;
  }

  async resizeWithFit(param: { width?: number; height?: number }) {
    if (!this.canvas) {
      return;
    }

    let newWidth: number = 0,
      newHeight: number = 0;
    if (param.width && this.canvas.width > param.width) {
      newWidth = param.width ? param.width : this.canvas.width;
      newHeight = this.canvas.height * (newWidth / this.canvas.width);
    } else if (param.height && this.canvas.height > param.height) {
      newHeight = param.height ? param.height : this.canvas.height;
      newWidth = this.canvas.width * (newHeight / this.canvas.height);
    } else {
      return;
    }

    const newCanvas = document.createElement('canvas');
    const newContext = newCanvas.getContext('2d')!;
    newCanvas.width = newWidth;
    newCanvas.height = newHeight;
    newContext.drawImage(this.canvas, 0, 0, newWidth, newHeight);
    this.replaceCanvas(newCanvas);
  }

  private replaceCanvas(canvas: HTMLCanvasElement) {
    if (!this.canvas || !this.context) {
      this.canvas = canvas;
      this.context = this.canvas.getContext('2d')!;
      return;
    }

    this.canvas.width = 0;
    this.canvas.height = 0;
    delete this.canvas;
    delete this.context;

    this.canvas = canvas;
    this.context = this.canvas.getContext('2d')!;
  }

  async getDataUrl(
    mime: 'image/jpeg' | 'image/png' | 'image/webp' = 'image/jpeg',
    imageQuality?: number
  ) {
    if (!this.canvas) {
      return null;
    }

    if (mime === 'image/jpeg' || mime === 'image/webp') {
      return this.canvas.toDataURL(mime, imageQuality);
    } else {
      return this.canvas.toDataURL(mime);
    }
  }

  private hexColorCodeToRgba(color: string): {
    r: number;
    g: number;
    b: number;
    a?: number;
  } {
    const r = parseInt(color.substr(1, 2), 16);
    const g = parseInt(color.substr(3, 2), 16);
    const b = parseInt(color.substr(5, 2), 16);
    if (color.length === 9) {
      const a = parseInt(color.substr(7, 2), 16);
      return { r, g, b, a };
    }
    return { r, g, b };
  }

  private rgbToHexColorCode(r: number, g: number, b: number) {
    return '#' + this.valueToHex(r) + this.valueToHex(g) + this.valueToHex(b);
  }

  private valueToHex(value: number) {
    return ('0' + value.toString(16)).slice(-2);
  }

  private isSimilarColor(
    color1: string,
    color2: string,
    diffThreshold: number
  ) {
    const color1Rgb = this.hexColorCodeToRgba(color1);
    const color2Rgb = this.hexColorCodeToRgba(color2);
    const diff =
      Math.abs(color1Rgb.r - color2Rgb.r) +
      Math.abs(color1Rgb.g - color2Rgb.g) +
      Math.abs(color1Rgb.b - color2Rgb.b);
    return diff < diffThreshold;
  }
}
