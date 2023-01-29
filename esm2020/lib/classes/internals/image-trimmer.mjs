export class ImageTrimmer {
    constructor() { }
    async loadByDataUrl(dataUrl) {
        const image = await new Promise((resolve, reject) => {
            const image = new Image();
            image.src = dataUrl;
            image.onload = () => {
                resolve(image);
            };
        });
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = image.width;
        canvas.height = image.height;
        context.drawImage(image, 0, 0);
        this.canvas = canvas;
        this.context = context;
    }
    async trimMargin(marginColor) {
        if (this.canvas === undefined)
            throw new Error('Image is not loaded');
        // マージンを検出する範囲を指定 (左端から0〜20%)
        const edgeDetectionRangeMinX = 0;
        const edgeDetectionRangeMaxX = Math.floor(this.canvas.width * 0.2);
        // マージンの端を検出
        const edgePositionFromTop = await this.getVerticalEdgePositionOfColor(marginColor, 'top', edgeDetectionRangeMinX, edgeDetectionRangeMaxX);
        const marginTop = edgePositionFromTop != null ? edgePositionFromTop : 0;
        const edgePositionFromBottom = await this.getVerticalEdgePositionOfColor(marginColor, 'bottom', edgeDetectionRangeMinX, edgeDetectionRangeMaxX);
        const marginBottom = edgePositionFromBottom != null
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
    async crop(x, y, w, h) {
        if (!this.canvas || !this.context)
            return;
        const newCanvas = document.createElement('canvas');
        const newContext = newCanvas.getContext('2d');
        newCanvas.width = w;
        newCanvas.height = h;
        newContext.drawImage(this.canvas, x, y, newCanvas.width, newCanvas.height, 0, 0, newCanvas.width, newCanvas.height);
        this.replaceCanvas(newCanvas);
    }
    async replaceColor(srcColor, dstColor, diffThreshold) {
        if (!this.canvas || !this.context)
            return;
        const imageData = this.context.getImageData(0, 0, this.canvas.width, this.canvas.height);
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
        let marginColor = null;
        const imageData = this.context.getImageData(0, 0, this.canvas.width, this.canvas.height);
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
                    }
                    else {
                        isBreak = true;
                        break;
                    }
                }
            }
        }
        return marginColor;
    }
    async getVerticalEdgePositionOfColor(color, direction, minX, maxX) {
        if (!this.canvas || !this.context) {
            return null;
        }
        if (minX === undefined) {
            minX = 0;
        }
        if (maxX === undefined) {
            maxX = this.canvas.width;
        }
        const imageData = this.context.getImageData(0, 0, this.canvas.width, this.canvas.height);
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
                    if (color == colorCode) {
                        if (edgePositionY < y) {
                            edgePositionY = y;
                        }
                    }
                    else {
                        isBreak = true;
                        break;
                    }
                }
            }
            if (edgePositionY !== 0) {
                edgePositionY += 1;
            }
        }
        else if (direction === 'bottom') {
            edgePositionY = this.canvas.height;
            for (let y = imageData.height - 1; y >= 0; y--) {
                for (let x = 0; x < imageData.width && !isBreak; x++) {
                    const idx = (x + y * imageData.width) * 4;
                    const red = imageData.data[idx + 0];
                    const green = imageData.data[idx + 1];
                    const blue = imageData.data[idx + 2];
                    const alpha = imageData.data[idx + 3];
                    const colorCode = this.rgbToHexColorCode(red, green, blue);
                    if (color == colorCode) {
                        if (edgePositionY > y) {
                            edgePositionY = y;
                        }
                    }
                    else {
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
    async resizeWithFit(param) {
        if (!this.canvas) {
            return;
        }
        let newWidth = 0, newHeight = 0;
        if (param.width && this.canvas.width > param.width) {
            newWidth = param.width ? param.width : this.canvas.width;
            newHeight = this.canvas.height * (newWidth / this.canvas.width);
        }
        else if (param.height && this.canvas.height > param.height) {
            newHeight = param.height ? param.height : this.canvas.height;
            newWidth = this.canvas.width * (newHeight / this.canvas.height);
        }
        else {
            return;
        }
        const newCanvas = document.createElement('canvas');
        const newContext = newCanvas.getContext('2d');
        newCanvas.width = newWidth;
        newCanvas.height = newHeight;
        newContext.drawImage(this.canvas, 0, 0, newWidth, newHeight);
        this.replaceCanvas(newCanvas);
    }
    replaceCanvas(canvas) {
        if (!this.canvas || !this.context) {
            this.canvas = canvas;
            this.context = this.canvas.getContext('2d');
            return;
        }
        this.canvas.width = 0;
        this.canvas.height = 0;
        delete this.canvas;
        delete this.context;
        this.canvas = canvas;
        this.context = this.canvas.getContext('2d');
    }
    async getDataUrl(mime = 'image/jpeg', imageQuality) {
        if (!this.canvas) {
            return null;
        }
        if (mime === 'image/jpeg' || mime === 'image/webp') {
            return this.canvas.toDataURL(mime, imageQuality);
        }
        else {
            return this.canvas.toDataURL(mime);
        }
    }
    hexColorCodeToRgba(color) {
        const r = parseInt(color.substr(1, 2), 16);
        const g = parseInt(color.substr(3, 2), 16);
        const b = parseInt(color.substr(5, 2), 16);
        if (color.length === 9) {
            const a = parseInt(color.substr(7, 2), 16);
            return { r, g, b, a };
        }
        return { r, g, b };
    }
    rgbToHexColorCode(r, g, b) {
        return '#' + this.valueToHex(r) + this.valueToHex(g) + this.valueToHex(b);
    }
    valueToHex(value) {
        return ('0' + value.toString(16)).slice(-2);
    }
    isSimilarColor(color1, color2, diffThreshold) {
        const color1Rgb = this.hexColorCodeToRgba(color1);
        const color2Rgb = this.hexColorCodeToRgba(color2);
        const diff = Math.abs(color1Rgb.r - color2Rgb.r) +
            Math.abs(color1Rgb.g - color2Rgb.g) +
            Math.abs(color1Rgb.b - color2Rgb.b);
        return diff < diffThreshold;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW1hZ2UtdHJpbW1lci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3Byb2plY3RzL25neC1tcC1wb3NlLWV4dHJhY3Rvci9zcmMvbGliL2NsYXNzZXMvaW50ZXJuYWxzL2ltYWdlLXRyaW1tZXIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBRUEsTUFBTSxPQUFPLFlBQVk7SUFJdkIsZ0JBQWUsQ0FBQztJQUVoQixLQUFLLENBQUMsYUFBYSxDQUFDLE9BQWU7UUFDakMsTUFBTSxLQUFLLEdBQXFCLE1BQU0sSUFBSSxPQUFPLENBQUMsQ0FBQyxPQUFPLEVBQUUsTUFBTSxFQUFFLEVBQUU7WUFDcEUsTUFBTSxLQUFLLEdBQUcsSUFBSSxLQUFLLEVBQUUsQ0FBQztZQUMxQixLQUFLLENBQUMsR0FBRyxHQUFHLE9BQU8sQ0FBQztZQUNwQixLQUFLLENBQUMsTUFBTSxHQUFHLEdBQUcsRUFBRTtnQkFDbEIsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDO1lBQ2pCLENBQUMsQ0FBQztRQUNKLENBQUMsQ0FBQyxDQUFDO1FBRUgsTUFBTSxNQUFNLEdBQUcsUUFBUSxDQUFDLGFBQWEsQ0FBQyxRQUFRLENBQUMsQ0FBQztRQUNoRCxNQUFNLE9BQU8sR0FBRyxNQUFNLENBQUMsVUFBVSxDQUFDLElBQUksQ0FBRSxDQUFDO1FBQ3pDLE1BQU0sQ0FBQyxLQUFLLEdBQUcsS0FBSyxDQUFDLEtBQUssQ0FBQztRQUMzQixNQUFNLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQyxNQUFNLENBQUM7UUFDN0IsT0FBTyxDQUFDLFNBQVMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBRS9CLElBQUksQ0FBQyxNQUFNLEdBQUcsTUFBTSxDQUFDO1FBQ3JCLElBQUksQ0FBQyxPQUFPLEdBQUcsT0FBTyxDQUFDO0lBQ3pCLENBQUM7SUFFRCxLQUFLLENBQUMsVUFBVSxDQUFDLFdBQW1CO1FBQ2xDLElBQUksSUFBSSxD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QVUsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsVUFBVSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzVFLENBQUM7SUFFTyxVQUFVLENBQUMsS0FBYTtRQUM5QixPQUFPLENBQUMsR0FBRyxHQUFHLEtBQUssQ0FBQyxRQUFRLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUM5QyxDQUFDO0lBRU8sY0FBYyxDQUNwQixNQUFjLEVBQ2QsTUFBYyxFQUNkLGFBQXFCO1FBRXJCLE1BQU0sU0FBUyxHQUFHLElBQUksQ0FBQyxrQkFBa0IsQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUNsRCxNQUFNLFNBQVMsR0FBRyxJQUFJLENBQUMsa0JBQWtCLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDbEQsTUFBTSxJQUFJLEdBQ1IsSUFBSSxDQUFDLEdBQUcsQ0FBQyxTQUFTLENBQUMsQ0FBQyxHQUFHLFNBQVMsQ0FBQyxDQUFDLENBQUM7WUFDbkMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxTQUFTLENBQUMsQ0FBQyxHQUFHLFNBQVMsQ0FBQyxDQUFDLENBQUM7WUFDbkMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxTQUFTLENBQUMsQ0FBQyxHQUFHLFNBQVMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN0QyxPQUFPLElBQUksR0FBRyxhQUFhLENBQUM7SUFDOUIsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHsgbHN0YXQgfSBmcm9tICdmcyc7XG5cbmV4cG9ydCBjbGFzcyBJbWFnZVRyaW1tZXIge1xuICBwcml2YXRlIGNhbnZhcz86IEhUTUxDYW52YXNFbGVtZW50O1xuICBwcml2YXRlIGNvbnRleHQ/OiBDYW52YXNSZW5kZXJpbmdDb250ZXh0MkQ7XG5cbiAgY29uc3RydWN0b3IoKSB7fVxuXG4gIGFzeW5jIGxvYWRCeURhdGFVcmwoZGF0YVVybDogc3RyaW5nKSB7XG4gICAgY29uc3QgaW1hZ2U6IEhUTUxJbWFnZUVsZW1lbnQgPSBhd2FpdCBuZXcgUHJvbWlzZSgocmVzb2x2ZSwgcmVqZWN0KSA9PiB7XG4gICAgICBjb25zdCBpbWFnZSA9IG5ldyBJbWFnZSgpO1xuICAgICAgaW1hZ2Uuc3JjID0gZGF0YVVybDtcbiAgICAgIGltYWdlLm9ubG9hZCA9ICgpID0+IHtcbiAgICAgICAgcmVzb2x2ZShpbWFnZSk7XG4gICAgICB9O1xuICAgIH0pO1xuXG4gICAgY29uc3QgY2FudmFzID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnY2FudmFzJyk7XG4gICAgY29uc3QgY29udGV4dCA9IGNhbnZhcy5nZXRDb250ZXh0KCcyZCcpITtcbiAgICBjYW52YXMud2lkdGggPSBpbWFnZS53aWR0aDtcbiAgICBjYW52YXMuaGVpZ2h0ID0gaW1hZ2UuaGVpZ2h0O1xuICAgIGNvbnRleHQuZHJhd0ltYWdlKGltYWdlLCAwLCAwKTtcblxuICAgIHRoaXMuY2FudmFzID0gY2FudmFzO1xuICAgIHRoaXMuY29udGV4dCA9IGNvbnRleHQ7XG4gIH1cblxuICBhc3luYyB0cmltTWFyZ2luKG1hcmdpbkNvbG9yOiBzdHJpbmcpIHtcbiAgICBpZiAodGhpcy5jYW52YXMgPT09IHVuZGVmaW5lZCkgdGhyb3cgbmV3IEVycm9yKCdJbWFnZSBpcyBub3QgbG9hZGVkJyk7XG5cbiAgICAvLyDjg57jg7zjgrjjg7PjgpLmpJzlh7rjgZnjgovnr4Tlm7LjgpLmjIflrpogKOW3puerr+OBi+OCiTDjgJwyMCUpXG4gICAgY29uc3QgZWRnZURldGVjdGlvblJhbmdlTWluWCA9IDA7XG4gICAgY29uc3QgZWRnZURldGVjdGlvblJhbmdlTWF4WCA9IE1hdGguZmxvb3IodGhpcy5jYW52YXMud2lkdGggKiAwLjIpO1xuXG4gICAgLy8g44Oe44O844K444Oz44Gu56uv44KS5qSc5Ye6XG4gICAgY29uc3QgZWRnZVBvc2l0aW9uRnJvbVRvcCA9IGF3YWl0IHRoaXMuZ2V0VmVydGljYWxFZGdlUG9zaXRpb25PZkNvbG9yKFxuICAgICAgbWFyZ2luQ29sb3IsXG4gICAgICAndG9wJyxcbiAgICAgIGVkZ2VEZXRlY3Rpb25SYW5nZU1pblgsXG4gICAgICBlZGdlRGV0ZWN0aW9uUmFuZ2VNYXhYXG4gICAgKTtcbiAgICBjb25zdCBtYXJnaW5Ub3AgPSBlZGdlUG9zaXRpb25Gcm9tVG9wICE9IG51bGwgPyBlZGdlUG9zaXRpb25Gcm9tVG9wIDogMDtcblxuICAgIGNvbnN0IGVkZ2VQb3NpdGlvbkZyb21Cb3R0b20gPSBhd2FpdCB0aGlzLmdldFZlcnRpY2FsRWRnZVBvc2l0aW9uT2ZDb2xvcihcbiAgICAgIG1hcmdpbkNvbG9yLFxuICAgICAgJ2JvdHRvbScsXG4gICAgICBlZGdlRGV0ZWN0aW9uUmFuZ2VNaW5YLFxuICAgICAgZWRnZURldGVjdGlvblJhbmdlTWF4WFxuICAgICk7XG4gICAgY29uc3QgbWFyZ2luQm90dG9tID1cbiAgICAgIGVkZ2VQb3NpdGlvbkZyb21Cb3R0b20gIT0gbnVsbFxuICAgICAgICA/IGVkZ2VQb3NpdGlvbkZyb21Cb3R0b21cbiAgICAgICAgOiB0aGlzLmNhbnZhcy5oZWlnaHQ7XG5cbiAgICBjb25zdCBvbGRIZWlnaHQgPSB0aGlzLmNhbnZhcy5oZWlnaHQ7XG4gICAgY29uc3QgbmV3SGVpZ2h0ID0gbWFyZ2luQm90dG9tIC0gbWFyZ2luVG9wO1xuICAgIHRoaXMuY3JvcCgwLCBtYXJnaW5Ub3AsIHRoaXMuY2FudmFzLndpZHRoLCBuZXdIZWlnaHQpO1xuXG4gICAgcmV0dXJuIHtcbiAgICAgIG1hcmdpblRvcDogbWFyZ2luVG9wLFxuICAgICAgbWFyZ2luQm90dG9tOiBtYXJnaW5Cb3R0b20sXG4gICAgICBoZWlnaHROZXc6IG5ld0hlaWdodCxcbiAgICAgIGhlaWdodE9sZDogb2xkSGVpZ2h0LFxuICAgICAgd2lkdGg6IHRoaXMuY2FudmFzLndpZHRoLFxuICAgIH07XG4gIH1cblxuICBhc3luYyBjcm9wKHg6IG51bWJlciwgeTogbnVtYmVyLCB3OiBudW1iZXIsIGg6IG51bWJlcikge1xuICAgIGlmICghdGhpcy5jYW52YXMgfHwgIXRoaXMuY29udGV4dCkgcmV0dXJuO1xuXG4gICAgY29uc3QgbmV3Q2FudmFzID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnY2FudmFzJyk7XG4gICAgY29uc3QgbmV3Q29udGV4dCA9IG5ld0NhbnZhcy5nZXRDb250ZXh0KCcyZCcpITtcbiAgICBuZXdDYW52YXMud2lkdGggPSB3O1xuICAgIG5ld0NhbnZhcy5oZWlnaHQgPSBoO1xuICAgIG5ld0NvbnRleHQuZHJhd0ltYWdlKFxuICAgICAgdGhpcy5jYW52YXMsXG4gICAgICB4LFxuICAgICAgeSxcbiAgICAgIG5ld0NhbnZhcy53aWR0aCxcbiAgICAgIG5ld0NhbnZhcy5oZWlnaHQsXG4gICAgICAwLFxuICAgICAgMCxcbiAgICAgIG5ld0NhbnZhcy53aWR0aCxcbiAgICAgIG5ld0NhbnZhcy5oZWlnaHRcbiAgICApO1xuICAgIHRoaXMucmVwbGFjZUNhbnZhcyhuZXdDYW52YXMpO1xuICB9XG5cbiAgYXN5bmMgcmVwbGFjZUNvbG9yKFxuICAgIHNyY0NvbG9yOiBzdHJpbmcsXG4gICAgZHN0Q29sb3I6IHN0cmluZyxcbiAgICBkaWZmVGhyZXNob2xkOiBudW1iZXJcbiAgKSB7XG4gICAgaWYgKCF0aGlzLmNhbnZhcyB8fCAhdGhpcy5jb250ZXh0KSByZXR1cm47XG5cbiAgICBjb25zdCBpbWFnZURhdGEgPSB0aGlzLmNvbnRleHQuZ2V0SW1hZ2VEYXRhKFxuICAgICAgMCxcbiAgICAgIDAsXG4gICAgICB0aGlzLmNhbnZhcy53aWR0aCxcbiAgICAgIHRoaXMuY2FudmFzLmhlaWdodFxuICAgICk7XG5cbiAgICBjb25zdCBkc3RDb2xvclZhbHVlID0gdGhpcy5oZXhDb2xvckNvZGVUb1JnYmEoZHN0Q29sb3IpO1xuXG4gICAgZm9yIChsZXQgeCA9IDA7IHggPCBpbWFnZURhdGEud2lkdGg7IHgrKykge1xuICAgICAgZm9yIChsZXQgeSA9IDA7IHkgPCBpbWFnZURhdGEuaGVpZ2h0OyB5KyspIHtcbiAgICAgICAgY29uc3QgaWR4ID0gKHggKyB5ICogaW1hZ2VEYXRhLndpZHRoKSAqIDQ7XG4gICAgICAgIGNvbnN0IHJlZCA9IGltYWdlRGF0YS5kYXRhW2lkeCArIDBdO1xuICAgICAgICBjb25zdCBncmVlbiA9IGltYWdlRGF0YS5kYXRhW2lkeCArIDFdO1xuICAgICAgICBjb25zdCBibHVlID0gaW1hZ2VEYXRhLmRhdGFbaWR4ICsgMl07XG4gICAgICAgIGNvbnN0IGFscGhhID0gaW1hZ2VEYXRhLmRhdGFbaWR4ICsgM107XG5cbiAgICAgICAgY29uc3QgY29sb3JDb2RlID0gdGhpcy5yZ2JUb0hleENvbG9yQ29kZShyZWQsIGdyZWVuLCBibHVlKTtcbiAgICAgICAgaWYgKCF0aGlzLmlzU2ltaWxhckNvbG9yKHNyY0NvbG9yLCBjb2xvckNvZGUsIGRpZmZUaHJlc2hvbGQpKSB7XG4gICAgICAgICAgY29udGludWU7XG4gICAgICAgIH1cblxuICAgICAgICBpbWFnZURhdGEuZGF0YVtpZHggKyAwXSA9IGRzdENvbG9yVmFsdWUucjtcbiAgICAgICAgaW1hZ2VEYXRhLmRhdGFbaWR4ICsgMV0gPSBkc3RDb2xvclZhbHVlLmc7XG4gICAgICAgIGltYWdlRGF0YS5kYXRhW2lkeCArIDJdID0gZHN0Q29sb3JWYWx1ZS5iO1xuICAgICAgICBpZiAoZHN0Q29sb3JWYWx1ZS5hICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICBpbWFnZURhdGEuZGF0YVtpZHggKyAzXSA9IGRzdENvbG9yVmFsdWUuYTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cblxuICAgIHRoaXMuY29udGV4dC5wdXRJbWFnZURhdGEoaW1hZ2VEYXRhLCAwLCAwKTtcbiAgfVxuXG4gIGFzeW5jIGdldE1hcmdpbkNvbG9yKCkge1xuICAgIGlmICghdGhpcy5jYW52YXMgfHwgIXRoaXMuY29udGV4dCkge1xuICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuICAgIGxldCBtYXJnaW5Db2xvcjogc3RyaW5nIHwgbnVsbCA9IG51bGw7XG5cbiAgICBjb25zdCBpbWFnZURhdGEgPSB0aGlzLmNvbnRleHQuZ2V0SW1hZ2VEYXRhKFxuICAgICAgMCxcbiAgICAgIDAsXG4gICAgICB0aGlzLmNhbnZhcy53aWR0aCxcbiAgICAgIHRoaXMuY2FudmFzLmhlaWdodFxuICAgICk7XG5cbiAgICBsZXQgaXNCcmVhayA9IGZhbHNlO1xuICAgIGZvciAobGV0IHggPSAwOyB4IDwgaW1hZ2VEYXRhLndpZHRoICYmICFpc0JyZWFrOyB4KyspIHtcbiAgICAgIGZvciAobGV0IHkgPSAwOyB5IDwgaW1hZ2VEYXRhLmhlaWdodDsgeSsrKSB7XG4gICAgICAgIGNvbnN0IGlkeCA9ICh4ICsgeSAqIGltYWdlRGF0YS53aWR0aCkgKiA0O1xuICAgICAgICBjb25zdCByZWQgPSBpbWFnZURhdGEuZGF0YVtpZHggKyAwXTtcbiAgICAgICAgY29uc3QgZ3JlZW4gPSBpbWFnZURhdGEuZGF0YVtpZHggKyAxXTtcbiAgICAgICAgY29uc3QgYmx1ZSA9IGltYWdlRGF0YS5kYXRhW2lkeCArIDJdO1xuICAgICAgICBjb25zdCBhbHBoYSA9IGltYWdlRGF0YS5kYXRhW2lkeCArIDNdO1xuXG4gICAgICAgIGNvbnN0IGNvbG9yQ29kZSA9IHRoaXMucmdiVG9IZXhDb2xvckNvZGUocmVkLCBncmVlbiwgYmx1ZSk7XG4gICAgICAgIGlmIChtYXJnaW5Db2xvciAhPSBjb2xvckNvZGUpIHtcbiAgICAgICAgICBpZiAobWFyZ2luQ29sb3IgPT09IG51bGwpIHtcbiAgICAgICAgICAgIG1hcmdpbkNvbG9yID0gY29sb3JDb2RlO1xuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBpc0JyZWFrID0gdHJ1ZTtcbiAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiBtYXJnaW5Db2xvcjtcbiAgfVxuXG4gIGFzeW5jIGdldFZlcnRpY2FsRWRnZVBvc2l0aW9uT2ZDb2xvcihcbiAgICBjb2xvcjogc3RyaW5nLFxuICAgIGRpcmVjdGlvbjogJ3RvcCcgfCAnYm90dG9tJyxcbiAgICBtaW5YPzogbnVtYmVyLFxuICAgIG1heFg/OiBudW1iZXJcbiAgKSB7XG4gICAgaWYgKCF0aGlzLmNhbnZhcyB8fCAhdGhpcy5jb250ZXh0KSB7XG4gICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG5cbiAgICBpZiAobWluWCA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICBtaW5YID0gMDtcbiAgICB9XG4gICAgaWYgKG1heFggPT09IHVuZGVmaW5lZCkge1xuICAgICAgbWF4WCA9IHRoaXMuY2FudmFzLndpZHRoO1xuICAgIH1cblxuICAgIGNvbnN0IGltYWdlRGF0YSA9IHRoaXMuY29udGV4dC5nZXRJbWFnZURhdGEoXG4gICAgICAwLFxuICAgICAgMCxcbiAgICAgIHRoaXMuY2FudmFzLndpZHRoLFxuICAgICAgdGhpcy5jYW52YXMuaGVpZ2h0XG4gICAgKTtcblxuICAgIGxldCBpc0JyZWFrID0gZmFsc2U7XG5cbiAgICBsZXQgZWRnZVBvc2l0aW9uWTtcbiAgICBpZiAoZGlyZWN0aW9uID09PSAndG9wJykge1xuICAgICAgZWRnZVBvc2l0aW9uWSA9IDA7XG5cbiAgICAgIGZvciAobGV0IHkgPSAwOyB5IDwgaW1hZ2VEYXRhLmhlaWdodDsgeSsrKSB7XG4gICAgICAgIGZvciAobGV0IHggPSAwOyB4IDwgbWF4WCAmJiAhaXNCcmVhazsgeCsrKSB7XG4gICAgICAgICAgY29uc3QgaWR4ID0gKHggKyB5ICogaW1hZ2VEYXRhLndpZHRoKSAqIDQ7XG4gICAgICAgICAgY29uc3QgcmVkID0gaW1hZ2VEYXRhLmRhdGFbaWR4ICsgMF07XG4gICAgICAgICAgY29uc3QgZ3JlZW4gPSBpbWFnZURhdGEuZGF0YVtpZHggKyAxXTtcbiAgICAgICAgICBjb25zdCBibHVlID0gaW1hZ2VEYXRhLmRhdGFbaWR4ICsgMl07XG4gICAgICAgICAgY29uc3QgYWxwaGEgPSBpbWFnZURhdGEuZGF0YVtpZHggKyAzXTtcblxuICAgICAgICAgIGNvbnN0IGNvbG9yQ29kZSA9IHRoaXMucmdiVG9IZXhDb2xvckNvZGUocmVkLCBncmVlbiwgYmx1ZSk7XG4gICAgICAgICAgaWYgKGNvbG9yID09IGNvbG9yQ29kZSkge1xuICAgICAgICAgICAgaWYgKGVkZ2VQb3NpdGlvblkgPCB5KSB7XG4gICAgICAgICAgICAgIGVkZ2VQb3NpdGlvblkgPSB5O1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBpc0JyZWFrID0gdHJ1ZTtcbiAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICBpZiAoZWRnZVBvc2l0aW9uWSAhPT0gMCkge1xuICAgICAgICBlZGdlUG9zaXRpb25ZICs9IDE7XG4gICAgICB9XG4gICAgfSBlbHNlIGlmIChkaXJlY3Rpb24gPT09ICdib3R0b20nKSB7XG4gICAgICBlZGdlUG9zaXRpb25ZID0gdGhpcy5jYW52YXMuaGVpZ2h0O1xuXG4gICAgICBmb3IgKGxldCB5ID0gaW1hZ2VEYXRhLmhlaWdodCAtIDE7IHkgPj0gMDsgeS0tKSB7XG4gICAgICAgIGZvciAobGV0IHggPSAwOyB4IDwgaW1hZ2VEYXRhLndpZHRoICYmICFpc0JyZWFrOyB4KyspIHtcbiAgICAgICAgICBjb25zdCBpZHggPSAoeCArIHkgKiBpbWFnZURhdGEud2lkdGgpICogNDtcbiAgICAgICAgICBjb25zdCByZWQgPSBpbWFnZURhdGEuZGF0YVtpZHggKyAwXTtcbiAgICAgICAgICBjb25zdCBncmVlbiA9IGltYWdlRGF0YS5kYXRhW2lkeCArIDFdO1xuICAgICAgICAgIGNvbnN0IGJsdWUgPSBpbWFnZURhdGEuZGF0YVtpZHggKyAyXTtcbiAgICAgICAgICBjb25zdCBhbHBoYSA9IGltYWdlRGF0YS5kYXRhW2lkeCArIDNdO1xuXG4gICAgICAgICAgY29uc3QgY29sb3JDb2RlID0gdGhpcy5yZ2JUb0hleENvbG9yQ29kZShyZWQsIGdyZWVuLCBibHVlKTtcbiAgICAgICAgICBpZiAoY29sb3IgPT0gY29sb3JDb2RlKSB7XG4gICAgICAgICAgICBpZiAoZWRnZVBvc2l0aW9uWSA+IHkpIHtcbiAgICAgICAgICAgICAgZWRnZVBvc2l0aW9uWSA9IHk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIGlzQnJlYWsgPSB0cnVlO1xuICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmIChlZGdlUG9zaXRpb25ZICE9PSB0aGlzLmNhbnZhcy5oZWlnaHQpIHtcbiAgICAgICAgZWRnZVBvc2l0aW9uWSAtPSAxO1xuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiBlZGdlUG9zaXRpb25ZO1xuICB9XG5cbiAgYXN5bmMgZ2V0V2lkdGgoKSB7XG4gICAgcmV0dXJuIHRoaXMuY2FudmFzPy53aWR0aDtcbiAgfVxuXG4gIGFzeW5jIGdldEhlaWdodCgpIHtcbiAgICByZXR1cm4gdGhpcy5jYW52YXM/LmhlaWdodDtcbiAgfVxuXG4gIGFzeW5jIHJlc2l6ZVdpdGhGaXQocGFyYW06IHsgd2lkdGg/OiBudW1iZXI7IGhlaWdodD86IG51bWJlciB9KSB7XG4gICAgaWYgKCF0aGlzLmNhbnZhcykge1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGxldCBuZXdXaWR0aDogbnVtYmVyID0gMCxcbiAgICAgIG5ld0hlaWdodDogbnVtYmVyID0gMDtcbiAgICBpZiAocGFyYW0ud2lkdGggJiYgdGhpcy5jYW52YXMud2lkdGggPiBwYXJhbS53aWR0aCkge1xuICAgICAgbmV3V2lkdGggPSBwYXJhbS53aWR0aCA/IHBhcmFtLndpZHRoIDogdGhpcy5jYW52YXMud2lkdGg7XG4gICAgICBuZXdIZWlnaHQgPSB0aGlzLmNhbnZhcy5oZWlnaHQgKiAobmV3V2lkdGggLyB0aGlzLmNhbnZhcy53aWR0aCk7XG4gICAgfSBlbHNlIGlmIChwYXJhbS5oZWlnaHQgJiYgdGhpcy5jYW52YXMuaGVpZ2h0ID4gcGFyYW0uaGVpZ2h0KSB7XG4gICAgICBuZXdIZWlnaHQgPSBwYXJhbS5oZWlnaHQgPyBwYXJhbS5oZWlnaHQgOiB0aGlzLmNhbnZhcy5oZWlnaHQ7XG4gICAgICBuZXdXaWR0aCA9IHRoaXMuY2FudmFzLndpZHRoICogKG5ld0hlaWdodCAvIHRoaXMuY2FudmFzLmhlaWdodCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICBjb25zdCBuZXdDYW52YXMgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdjYW52YXMnKTtcbiAgICBjb25zdCBuZXdDb250ZXh0ID0gbmV3Q2FudmFzLmdldENvbnRleHQoJzJkJykhO1xuICAgIG5ld0NhbnZhcy53aWR0aCA9IG5ld1dpZHRoO1xuICAgIG5ld0NhbnZhcy5oZWlnaHQgPSBuZXdIZWlnaHQ7XG4gICAgbmV3Q29udGV4dC5kcmF3SW1hZ2UodGhpcy5jYW52YXMsIDAsIDAsIG5ld1dpZHRoLCBuZXdIZWlnaHQpO1xuICAgIHRoaXMucmVwbGFjZUNhbnZhcyhuZXdDYW52YXMpO1xuICB9XG5cbiAgcHJpdmF0ZSByZXBsYWNlQ2FudmFzKGNhbnZhczogSFRNTENhbnZhc0VsZW1lbnQpIHtcbiAgICBpZiAoIXRoaXMuY2FudmFzIHx8ICF0aGlzLmNvbnRleHQpIHtcbiAgICAgIHRoaXMuY2FudmFzID0gY2FudmFzO1xuICAgICAgdGhpcy5jb250ZXh0ID0gdGhpcy5jYW52YXMuZ2V0Q29udGV4dCgnMmQnKSE7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgdGhpcy5jYW52YXMud2lkdGggPSAwO1xuICAgIHRoaXMuY2FudmFzLmhlaWdodCA9IDA7XG4gICAgZGVsZXRlIHRoaXMuY2FudmFzO1xuICAgIGRlbGV0ZSB0aGlzLmNvbnRleHQ7XG5cbiAgICB0aGlzLmNhbnZhcyA9IGNhbnZhcztcbiAgICB0aGlzLmNvbnRleHQgPSB0aGlzLmNhbnZhcy5nZXRDb250ZXh0KCcyZCcpITtcbiAgfVxuXG4gIGFzeW5jIGdldERhdGFVcmwoXG4gICAgbWltZTogJ2ltYWdlL2pwZWcnIHwgJ2ltYWdlL3BuZycgfCAnaW1hZ2Uvd2VicCcgPSAnaW1hZ2UvanBlZycsXG4gICAgaW1hZ2VRdWFsaXR5PzogbnVtYmVyXG4gICkge1xuICAgIGlmICghdGhpcy5jYW52YXMpIHtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH1cblxuICAgIGlmIChtaW1lID09PSAnaW1hZ2UvanBlZycgfHwgbWltZSA9PT0gJ2ltYWdlL3dlYnAnKSB7XG4gICAgICByZXR1cm4gdGhpcy5jYW52YXMudG9EYXRhVVJMKG1pbWUsIGltYWdlUXVhbGl0eSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVybiB0aGlzLmNhbnZhcy50b0RhdGFVUkwobWltZSk7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBoZXhDb2xvckNvZGVUb1JnYmEoY29sb3I6IHN0cmluZyk6IHtcbiAgICByOiBudW1iZXI7XG4gICAgZzogbnVtYmVyO1xuICAgIGI6IG51bWJlcjtcbiAgICBhPzogbnVtYmVyO1xuICB9IHtcbiAgICBjb25zdCByID0gcGFyc2VJbnQoY29sb3Iuc3Vic3RyKDEsIDIpLCAxNik7XG4gICAgY29uc3QgZyA9IHBhcnNlSW50KGNvbG9yLnN1YnN0cigzLCAyKSwgMTYpO1xuICAgIGNvbnN0IGIgPSBwYXJzZUludChjb2xvci5zdWJzdHIoNSwgMiksIDE2KTtcbiAgICBpZiAoY29sb3IubGVuZ3RoID09PSA5KSB7XG4gICAgICBjb25zdCBhID0gcGFyc2VJbnQoY29sb3Iuc3Vic3RyKDcsIDIpLCAxNik7XG4gICAgICByZXR1cm4geyByLCBnLCBiLCBhIH07XG4gICAgfVxuICAgIHJldHVybiB7IHIsIGcsIGIgfTtcbiAgfVxuXG4gIHByaXZhdGUgcmdiVG9IZXhDb2xvckNvZGUocjogbnVtYmVyLCBnOiBudW1iZXIsIGI6IG51bWJlcikge1xuICAgIHJldHVybiAnIycgKyB0aGlzLnZhbHVlVG9IZXgocikgKyB0aGlzLnZhbHVlVG9IZXgoZykgKyB0aGlzLnZhbHVlVG9IZXgoYik7XG4gIH1cblxuICBwcml2YXRlIHZhbHVlVG9IZXgodmFsdWU6IG51bWJlcikge1xuICAgIHJldHVybiAoJzAnICsgdmFsdWUudG9TdHJpbmcoMTYpKS5zbGljZSgtMik7XG4gIH1cblxuICBwcml2YXRlIGlzU2ltaWxhckNvbG9yKFxuICAgIGNvbG9yMTogc3RyaW5nLFxuICAgIGNvbG9yMjogc3RyaW5nLFxuICAgIGRpZmZUaHJlc2hvbGQ6IG51bWJlclxuICApIHtcbiAgICBjb25zdCBjb2xvcjFSZ2IgPSB0aGlzLmhleENvbG9yQ29kZVRvUmdiYShjb2xvcjEpO1xuICAgIGNvbnN0IGNvbG9yMlJnYiA9IHRoaXMuaGV4Q29sb3JDb2RlVG9SZ2JhKGNvbG9yMik7XG4gICAgY29uc3QgZGlmZiA9XG4gICAgICBNYXRoLmFicyhjb2xvcjFSZ2IuciAtIGNvbG9yMlJnYi5yKSArXG4gICAgICBNYXRoLmFicyhjb2xvcjFSZ2IuZyAtIGNvbG9yMlJnYi5nKSArXG4gICAgICBNYXRoLmFicyhjb2xvcjFSZ2IuYiAtIGNvbG9yMlJnYi5iKTtcbiAgICByZXR1cm4gZGlmZiA8IGRpZmZUaHJlc2hvbGQ7XG4gIH1cbn1cbiJdfQ==