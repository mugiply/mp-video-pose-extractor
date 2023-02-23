export declare class ImageTrimmer {
    private canvas?;
    private context?;
    constructor();
    loadByDataUrl(dataUrl: string): Promise<void>;
    trimMargin(marginColor: string, diffThreshold?: number): Promise<{
        marginTop: number;
        marginBottom: number;
        heightNew: number;
        heightOld: number;
        width: number;
    }>;
    crop(x: number, y: number, w: number, h: number): Promise<void>;
    replaceColor(srcColor: string, dstColor: string, diffThreshold: number): Promise<void>;
    getMarginColor(): Promise<string>;
    getVerticalEdgePositionOfColor(color: string, direction: 'top' | 'bottom', diffThreshold: number, minX?: number, maxX?: number): Promise<number>;
    getWidth(): Promise<number>;
    getHeight(): Promise<number>;
    resizeWithFit(param: {
        width?: number;
        height?: number;
    }): Promise<void>;
    private replaceCanvas;
    getDataUrl(mime?: 'image/jpeg' | 'image/png' | 'image/webp', imageQuality?: number): Promise<string>;
    private hexColorCodeToRgba;
    private rgbToHexColorCode;
    private valueToHex;
    private isSimilarColor;
}
