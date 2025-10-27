export namespace main {
	
	export class ArgumentDefinition {
	    flag: string;
	    type: string;
	    socketSide: string;
	    isRequired: boolean;
	    defaultValue: string;
	    description: string;
	    validation: string;
	    userOverride: boolean;
	    skipEmit?: boolean;
	
	    static createFrom(source: any = {}) {
	        return new ArgumentDefinition(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.flag = source["flag"];
	        this.type = source["type"];
	        this.socketSide = source["socketSide"];
	        this.isRequired = source["isRequired"];
	        this.defaultValue = source["defaultValue"];
	        this.description = source["description"];
	        this.validation = source["validation"];
	        this.userOverride = source["userOverride"];
	        this.skipEmit = source["skipEmit"];
	    }
	}
	export class CLIDefinition {
	    id: string;
	    name: string;
	    category: string;
	    icon: string;
	    color: string;
	    description: string;
	    environment: string;
	    executable: string;
	    script: string;
	    helpCommand: string;
	    arguments: ArgumentDefinition[];
	    version: string;
	    author: string;
	    lastParsed?: string;
	
	    static createFrom(source: any = {}) {
	        return new CLIDefinition(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.name = source["name"];
	        this.category = source["category"];
	        this.icon = source["icon"];
	        this.color = source["color"];
	        this.description = source["description"];
	        this.environment = source["environment"];
	        this.executable = source["executable"];
	        this.script = source["script"];
	        this.helpCommand = source["helpCommand"];
	        this.arguments = this.convertValues(source["arguments"], ArgumentDefinition);
	        this.version = source["version"];
	        this.author = source["author"];
	        this.lastParsed = source["lastParsed"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class Socket {
	    id: string;
	    nodeId: string;
	    argumentFlag: string;
	    type: string;
	    socketSide: string;
	    value: string;
	    connectedTo?: string;
	    isRequired: boolean;
	    defaultValue: string;
	    description: string;
	    validation: string;
	    skipEmit?: boolean;
	
	    static createFrom(source: any = {}) {
	        return new Socket(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.nodeId = source["nodeId"];
	        this.argumentFlag = source["argumentFlag"];
	        this.type = source["type"];
	        this.socketSide = source["socketSide"];
	        this.value = source["value"];
	        this.connectedTo = source["connectedTo"];
	        this.isRequired = source["isRequired"];
	        this.defaultValue = source["defaultValue"];
	        this.description = source["description"];
	        this.validation = source["validation"];
	        this.skipEmit = source["skipEmit"];
	    }
	}
	export class Size {
	    width: number;
	    height: number;
	
	    static createFrom(source: any = {}) {
	        return new Size(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.width = source["width"];
	        this.height = source["height"];
	    }
	}
	export class Point {
	    x: number;
	    y: number;
	
	    static createFrom(source: any = {}) {
	        return new Point(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.x = source["x"];
	        this.y = source["y"];
	    }
	}
	export class CLINode {
	    id: string;
	    definitionId: string;
	    name: string;
	    position: Point;
	    size: Size;
	    environment: string;
	    executable: string;
	    script: string;
	    inputSockets: Socket[];
	    outputSockets: Socket[];
	    icon: string;
	    color: string;
	    isSelected: boolean;
	    isCollapsed: boolean;
	    category: string;
	    testStatus: string;
	    lastTestFile: string;
	    lastTestOutput: string;
	    lastTestTime: string;
	    testError: string;
	
	    static createFrom(source: any = {}) {
	        return new CLINode(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.definitionId = source["definitionId"];
	        this.name = source["name"];
	        this.position = this.convertValues(source["position"], Point);
	        this.size = this.convertValues(source["size"], Size);
	        this.environment = source["environment"];
	        this.executable = source["executable"];
	        this.script = source["script"];
	        this.inputSockets = this.convertValues(source["inputSockets"], Socket);
	        this.outputSockets = this.convertValues(source["outputSockets"], Socket);
	        this.icon = source["icon"];
	        this.color = source["color"];
	        this.isSelected = source["isSelected"];
	        this.isCollapsed = source["isCollapsed"];
	        this.category = source["category"];
	        this.testStatus = source["testStatus"];
	        this.lastTestFile = source["lastTestFile"];
	        this.lastTestOutput = source["lastTestOutput"];
	        this.lastTestTime = source["lastTestTime"];
	        this.testError = source["testError"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class FileListResult {
	    files: string[];
	    count: number;
	    error?: string;
	
	    static createFrom(source: any = {}) {
	        return new FileListResult(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.files = source["files"];
	        this.count = source["count"];
	        this.error = source["error"];
	    }
	}
	export class PathToken {
	    token: string;
	    description: string;
	    resolvedPath: string;
	
	    static createFrom(source: any = {}) {
	        return new PathToken(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.token = source["token"];
	        this.description = source["description"];
	        this.resolvedPath = source["resolvedPath"];
	    }
	}
	export class PipelineMetadata {
	    name: string;
	    description: string;
	    version: string;
	    author: string;
	    // Go type: time
	    created: any;
	    // Go type: time
	    modified: any;
	
	    static createFrom(source: any = {}) {
	        return new PipelineMetadata(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.name = source["name"];
	        this.description = source["description"];
	        this.version = source["version"];
	        this.author = source["author"];
	        this.created = this.convertValues(source["created"], null);
	        this.modified = this.convertValues(source["modified"], null);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class SocketConnection {
	    id: string;
	    fromNodeId: string;
	    fromSocketId: string;
	    toNodeId: string;
	    toSocketId: string;
	    isValid: boolean;
	
	    static createFrom(source: any = {}) {
	        return new SocketConnection(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.fromNodeId = source["fromNodeId"];
	        this.fromSocketId = source["fromSocketId"];
	        this.toNodeId = source["toNodeId"];
	        this.toSocketId = source["toSocketId"];
	        this.isValid = source["isValid"];
	    }
	}
	export class Pipeline {
	    nodes: CLINode[];
	    connections: SocketConnection[];
	    metadata: PipelineMetadata;
	
	    static createFrom(source: any = {}) {
	        return new Pipeline(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.nodes = this.convertValues(source["nodes"], CLINode);
	        this.connections = this.convertValues(source["connections"], SocketConnection);
	        this.metadata = this.convertValues(source["metadata"], PipelineMetadata);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	
	
	
	

}

