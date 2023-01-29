export declare class ImageTrimmer {
    private canvas?;
    private context?;
    constructor();
    loadByDataUrl(dataUrl: string): Promise<void>;
    trimMargin(marginColor: string): Promise<{
        marginTop: number;
        marginBottom: number;
        heightNew: number;
        heightOld: number;
        width: number;
    }>;
    crop(x: number, y: number, w: number, h: number): Promise<void>;
    getMarginColor(): Promise<string | null>;
    getVerticalEdgePositionOfColor(color: string, direction: 'top' | 'bottom', minX?: number, maxX?: number): Promise<number | null | undefined>;
    getWidth(): Promise<number | undefined>;
    getHeight(): Promise<number | undefined>;
    resizeWithFit(param: {
        width?: number;
        height?: number;
    }): Promise<void>;
    private replaceCanvas;
    getDataUrl(mime?: 'image/jpeg' | 'image/png', jpegQuality?: number): Promise<string | null>;
    private rgbToHexColorCode;
    private valueToHex;
}
